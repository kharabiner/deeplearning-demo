"""
features/expand.py — Expand (프레임 확장 / 아웃페인팅)

iOS 27 "Expand" 재현: 사진 프레임을 축소해 바깥쪽 여백을 노출시키고,
그 바깥 여백을 자연스럽게 생성한다.

  - 분석(1회): LaMa 로 최대 확장 미리보기 배경 생성 (프롬프트 없음, 주변 맥락)
  - 드래그: LaMa 배경을 슬라이더와 같이 줌 + 원본 축소·페더 합성 (가벼움)
  - [완료]: DreamShaper SD1.5 inpaint — LaMa 맥락 캔버스

UI 흐름:
  expand_analyze → (슬라이더) expand_view → expand_commit
"""

from __future__ import annotations

import gradio as gr
import numpy as np
from PIL import Image

from common import pil_to_numpy, numpy_to_pil, resize_if_needed, free_memory
import inpaint as rinp
from task_inpaint_sd15 import MODEL_ID
from .shared import (
    DEVICE, PREVIEW_MAX, HIDDEN, VISIBLE,
    inpaint_commit, dilate_mask, feather_composite,
    SD15_PROMPT_EXPAND,
)

BACKDROP_SIGMA = 24.0   # LaMa 시드용 가우시안 블러(분석 1회)
FEATHER_FRAC = 0.025    # 선명/배경 경계 페더 폭(이미지 짧은 변 대비)
MAX_EXTEND = 1.6        # LaMa 미리보기 배경 생성 기준(슬라이더 최대와 일치)

# DreamShaper SD1.5 inpaint — Expand [완료] (고정 프롬프트)
EXPAND_SD15_PROMPT = SD15_PROMPT_EXPAND
EXPAND_SD15_GUIDANCE = 7.5
EXPAND_SD15_STEPS = 25
EXPAND_SD15_LONG = 512
EXPAND_SD15_NEGATIVE = (
    "frame, picture frame, photo border, canvas, mat, vignette, poster, "
    "new object, duplicate, extra limbs, "
    "blurry, distorted, artifacts, watermark, text, low quality, deformed"
)


def _make_backdrop(img_np: np.ndarray) -> np.ndarray:
    """프레임을 채울 블러 배경을 분석 때 1회 생성(이후 드래그 중 재사용).

    원본을 강하게 블러해 두면, 슬라이더로 원본이 작아질 때 노출되는 바깥
    여백이 '부드럽게 확장된' 듯 보인다.
    """
    try:
        import cv2
        return cv2.GaussianBlur(img_np, (0, 0), BACKDROP_SIGMA)
    except Exception:
        from PIL import ImageFilter
        return np.asarray(numpy_to_pil(img_np).filter(ImageFilter.GaussianBlur(BACKDROP_SIGMA)))


def _resize(img_np: np.ndarray, nw: int, nh: int, *, fast: bool) -> np.ndarray:
    """fast=True 면 드래그용 BILINEAR(가벼움), False 면 확정용 LANCZOS(고품질)."""
    if fast:
        try:
            import cv2
            return cv2.resize(img_np, (nw, nh), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return np.asarray(numpy_to_pil(img_np).resize((nw, nh), Image.BILINEAR))
    return np.asarray(numpy_to_pil(img_np).resize((nw, nh), Image.LANCZOS))


def _inner_alpha(H, W, x0, y0, nw, nh, feather):
    """선명 영역(중앙)=1, 바깥=0, 경계는 feather 폭으로 부드럽게 감소하는 알파."""
    inner = np.zeros((H, W), np.float32)
    inner[y0:y0 + nh, x0:x0 + nw] = 1.0
    if feather <= 0:
        return inner
    try:
        import cv2
        inner = cv2.GaussianBlur(inner, (0, 0), feather)
    except Exception:
        pass
    return np.clip(inner, 0.0, 1.0)


# ── 원본을 중앙 축소 배치 → 바깥 여백(outer) 노출 ───────────────────────────────
def _compose(img_np, extend, *, backdrop=None, fast=False, feather=True):
    """extend>1 일수록 원본을 더 작게 중앙 배치 → 바깥 여백이 넓어짐.

    backdrop 가 주어지면 바깥 여백을 그 블러 배경으로 채우고, 경계를 페더로
    부드럽게 섞는다(딱딱한 직사각형 씸 제거).
    Returns: (canvas uint8 (H,W,3), outer bool (H,W))  — outer 는 인페인팅 대상.
    """
    H, W = img_np.shape[:2]
    s = 1.0 / float(extend)
    nw, nh = max(1, round(W * s)), max(1, round(H * s))
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    fpx = max(2.0, FEATHER_FRAC * min(H, W)) if feather else 0.0

    base = backdrop if backdrop is not None else np.zeros((H, W, 3), np.uint8)
    sharp = base.copy()
    sharp[y0:y0 + nh, x0:x0 + nw] = _resize(img_np, nw, nh, fast=fast)

    a = _inner_alpha(H, W, x0, y0, nw, nh, fpx)
    if fpx > 0:
        a3 = a[..., None]
        canvas = (base.astype(np.float32) * (1.0 - a3)
                  + sharp.astype(np.float32) * a3).clip(0, 255).astype(np.uint8)
    else:
        canvas = sharp

    # 페더 밴드까지 인페인팅 대상에 포함 → 생성 경계가 자연스럽게 이어짐
    outer = a < 0.95
    return canvas, outer


def _zoom_center(img_np: np.ndarray, zoom: float, *, fast: bool = True) -> np.ndarray:
    """중심 기준 줌 — LaMa 배경을 슬라이더와 같이 확대/축소.

    zoom=1 그대로, zoom>1 중심 확대(가장자리 잘림) 후 캔버스 크기로 복원.
    extend 가 1.0→1.6 으로 갈수록 zoom 이 MAX_EXTEND→1 로 줄어 배경이 밖으로 펼쳐짐.
    """
    if zoom <= 1.0 + 1e-3:
        return img_np
    H, W = img_np.shape[:2]
    nh, nw = max(1, round(H / zoom)), max(1, round(W / zoom))
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    crop = img_np[y0:y0 + nh, x0:x0 + nw]
    return _resize(crop, W, H, fast=fast)


def _scaled_lama_backdrop(lama_backdrop: np.ndarray, extend: float, *, fast: bool) -> np.ndarray:
    """최대 확장(MAX_EXTEND) 기준 LaMa 결과를 현재 extend 에 맞게 중심 줌."""
    zoom = MAX_EXTEND / float(extend)
    return _zoom_center(lama_backdrop, zoom, fast=fast)


def _view(img_np, backdrop, extend):
    """드래그 미리보기: LaMa 배경을 슬라이더와 같이 줌 + 원본 축소·페더 합성."""
    if abs(float(extend) - 1.0) < 1e-3:
        return img_np
    if backdrop is None:
        backdrop = _make_backdrop(img_np)
    bd = _scaled_lama_backdrop(backdrop, extend, fast=True)
    canvas, _ = _compose(img_np, extend, backdrop=bd, fast=True)
    return canvas


def _lama_backdrop(img_np, progress) -> np.ndarray:
    """최대 확장 기준 LaMa 배경 1회 생성 → 드래그 미리보기·SD 맥락용."""
    blur = _make_backdrop(img_np)
    canvas, outer = _compose(
        img_np, MAX_EXTEND, backdrop=blur, fast=False, feather=False,
    )
    fill = dilate_mask(outer, iterations=2)
    lama_pil, _ = inpaint_commit(
        numpy_to_pil(canvas), fill, progress,
        desc="LaMa 미리보기", backend="lama",
    )
    return pil_to_numpy(lama_pil)


# ── 1) 분석 (LaMa만 — SD 로드 없음) ───────────────────────────────────────────
def expand_analyze(image, progress=gr.Progress()):
    if image is None:
        raise gr.Error("먼저 왼쪽에 사진을 업로드하세요.")
    rinp.unload_expand_sd15()
    from features.clean_up import unload_sam
    unload_sam()
    free_memory(DEVICE)
    progress(0.05, desc="Expand — 준비 중...")
    small = resize_if_needed(image.convert("RGB"), max_size=PREVIEW_MAX)
    img_np = pil_to_numpy(small)
    progress(0.15, desc="Expand — LaMa 미리보기 배경 생성")
    backdrop = _lama_backdrop(img_np, progress)
    return (
        None, None, backdrop, img_np, None, "expand",
        gr.update(value=img_np, visible=True),
        HIDDEN,
        "✅ Expand 준비 · LaMa 미리보기 → 슬라이더 조절 → [완료] DreamShaper SD1.5",
        HIDDEN, VISIBLE, HIDDEN,
        gr.skip(), gr.skip(),
    )


# ── 2) 미리보기 (LaMa 배경 + 축소 합성, 가벼움) ─────────────────────────────────
def expand_view(_disp, backdrop, img_np, extend):
    """드래그 중: LaMa 배경 위에 원본을 축소해 얹기만."""
    if img_np is None:
        return None
    return _view(img_np, backdrop, extend)


# ── 3) 확정 (DreamShaper SD1.5 inpaint) ───────────────────────────────────────
def expand_commit(_disp, backdrop, img_np, extend, progress=gr.Progress()):
    if img_np is None:
        raise gr.Error("먼저 Expand를 실행하세요.")
    if abs(float(extend) - 1.0) < 1e-3:
        return numpy_to_pil(img_np), "확장 없음 — 슬라이더로 프레임을 줄이세요."
    if backdrop is None:
        progress(0.1, desc="LaMa 배경 생성")
        backdrop = _lama_backdrop(img_np, progress)
    progress(0.2, desc="프레임 확장")
    bd = _scaled_lama_backdrop(backdrop, extend, fast=False)
    canvas, outer = _compose(img_np, extend, backdrop=bd, fast=False)
    fill = dilate_mask(outer, iterations=2)
    canvas_pil = numpy_to_pil(canvas)
    free_memory(DEVICE)
    prompt = EXPAND_SD15_PROMPT
    progress(0.35, desc="DreamShaper 로드")
    try:
        inp = rinp.preload_expand_sd15(DEVICE)
        progress(0.6, desc="아웃페인팅 — DreamShaper")
        result_pil = inp.inpaint(
            canvas_pil, fill,
            prompt=prompt,
            guidance=EXPAND_SD15_GUIDANCE,
            steps=EXPAND_SD15_STEPS,
            long=EXPAND_SD15_LONG,
            negative_prompt=EXPAND_SD15_NEGATIVE,
        )
    except Exception as e:
        raise gr.Error(f"아웃페인팅 실패(DreamShaper): {e}")
    result_pil = feather_composite(canvas_pil, result_pil, fill, feather=3.0)
    model_short = MODEL_ID.split("/")[-1]
    msg = (
        f"완료 · {model_short} {EXPAND_SD15_STEPS}step · "
        f"prompt={prompt[:40]} · 우상단 아이콘으로 다운로드"
    )
    return result_pil, msg
