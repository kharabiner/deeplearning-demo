"""
features/shared.py — 세 기능(clean up / expand / reframe) 공용 글루 코드

모델별 코드(task_*.py)와 기능별 코드(clean_up/expand/reframe) 사이의 얇은 중간층.
모델을 불러와 돌리고 결과를 이어붙이는, 여러 기능이 함께 쓰는 함수만 모은다.

  - DEVICE / 렌더 상수
  - HIDDEN / VISIBLE : Gradio update 헬퍼
  - vlm_caption / vlm_caption_reframe : Qwen2-VL → SD 인페인팅 프롬프트
  - inpaint_commit : SD2 인페인팅 실행 래퍼
  - feather_composite : 채운 영역을 원본에 부드럽게 합성
  - resize_mask : bool 마스크 크기 변경
"""

from __future__ import annotations

import gradio as gr
import numpy as np
from PIL import Image

from common import (
    get_device, free_memory, pil_to_numpy, numpy_to_pil,
)
import inpaint as rinp

DEVICE = get_device()

# ── 렌더/미리보기 상수 ──────────────────────────────────────────────────────────
PREVIEW_MAX = 1024         # 미리보기/격자 렌더 긴 변 (고해상도)
COMMIT_LONG = 1280         # 확정 렌더 긴 변 (최종 품질 우선)
RENDER_SS_COMMIT = 2       # SHARP 확정 렌더 슈퍼샘플 배수
FALLBACK_PARALLAX = 6.0    # depth 워핑 패럴랙스 (미리보기용)

# ── Gradio 업데이트 헬퍼 ────────────────────────────────────────────────────────
HIDDEN = gr.update(visible=False)
VISIBLE = gr.update(visible=True)


# ── VLM 캡션 (SD 인페인팅 프롬프트) ─────────────────────────────────────────────
def vlm_caption(image: Image.Image) -> str:
    """SD2 인페인팅 프롬프트용 캡션. 실패 시 기본 프롬프트."""
    try:
        import task_vlm_qwen2vl as task_vlm
        vproc, vmodel = task_vlm.load_model(DEVICE)
        desc = task_vlm.run(
            image, vproc, vmodel,
            question="Describe the background scene in a short phrase for image inpainting "
                     "(scene type, colors, lighting). Keep it under 15 words.",
            max_new_tokens=48, device=DEVICE,
        )
        del vproc, vmodel
        free_memory(DEVICE)
        return f"{desc.strip()}, {rinp.DEFAULT_PROMPT}"
    except Exception as e:
        print(f"[shared] VLM 캡션 실패 → 기본 프롬프트: {e}")
        return rinp.DEFAULT_PROMPT


def vlm_caption_reframe(image: Image.Image) -> str:
    """Reframe용 보수적 캡션 — 장면 전체 재생성이 아니라 가장자리 연장."""
    try:
        import task_vlm_qwen2vl as task_vlm
        vproc, vmodel = task_vlm.load_model(DEVICE)
        desc = task_vlm.run(
            image, vproc, vmodel,
            question="Describe only the background texture and colors at the image edges "
                     "for seamless extension. Under 12 words. Do not mention objects.",
            max_new_tokens=40, device=DEVICE,
        )
        del vproc, vmodel
        free_memory(DEVICE)
        return f"seamless extension of {desc.strip()}, same lighting, photorealistic"
    except Exception as e:
        print(f"[shared] Reframe VLM 캡션 실패 → 기본 프롬프트: {e}")
        return "seamless natural background extension, same scene, photorealistic"


# ── 인페인팅 실행 (확정 공용) ───────────────────────────────────────────────────
def inpaint_commit(
    image_pil, fill, progress, desc="인페인팅",
    prompt=None, caption_fn=None, backend="sd2", sd_guidance=7.5,
):
    """인페인팅 실행. (결과 PIL, 상태 메시지) 반환.

    backend="lama" : 프롬프트 없이 주변 맥락으로 채움 (Expand 아웃페인팅 권장)
    backend="sd2"  : SDXL 인페인팅 + 텍스트 프롬프트 (Clean Up 등)
    """
    if int(fill.sum()) < 30:
        return image_pil, "완료 · 채울 영역 없음"

    if backend == "lama":
        progress(0.6, desc=f"{desc} — LaMa")
        try:
            inp = rinp.get_inpainter("lama", DEVICE)
            result = inp.inpaint(image_pil, fill)
            inp.unload()
        except Exception as e:
            raise gr.Error(f"{desc} 실패(LaMa): {e}")
        return result, f"완료 · LaMa(프롬프트 없음, 주변 맥락) · 우상단 아이콘으로 다운로드"

    if prompt is None:
        prompt = (caption_fn or vlm_caption)(image_pil)
    progress(0.6, desc=f"{desc} — SDXL")
    try:
        inp = rinp.get_inpainter("sd2", DEVICE)
        result = inp.inpaint(image_pil, fill, prompt=prompt, guidance=sd_guidance)
        inp.unload()
    except Exception as e:
        raise gr.Error(f"{desc} 실패(SDXL): {e}")

    prompt_label = "(없음, 이미지 맥락)" if prompt == "" else prompt[:50]
    return result, f"완료 · SDXL · prompt={prompt_label} · 우상단 아이콘으로 다운로드"


# ── 합성/마스크 헬퍼 ────────────────────────────────────────────────────────────
def feather_composite(orig_pil, filled_pil, fill_mask, feather=2.5):
    """채운 영역을 원본에 부드러운 경계로 합성 → 하드 씸(딱딱한 테두리) 제거."""
    orig = pil_to_numpy(orig_pil).astype(np.float32)
    fill = np.asarray(filled_pil).astype(np.float32)
    if fill.shape != orig.shape:
        fill = pil_to_numpy(filled_pil.resize(orig_pil.size, Image.LANCZOS)).astype(np.float32)
    try:
        import cv2
        alpha = cv2.GaussianBlur(fill_mask.astype(np.float32), (0, 0), feather)
    except Exception:
        alpha = fill_mask.astype(np.float32)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    out = orig * (1.0 - alpha) + fill * alpha
    return numpy_to_pil(out.clip(0, 255).astype(np.uint8))


def resize_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """bool 마스크를 (H, W) 로 NEAREST 리사이즈."""
    if mask.shape[:2] == (H, W):
        return mask
    return np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
    ).astype(bool)


def dilate_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """인페인팅 경계를 깔끔하게 하려 마스크를 살짝 팽창 (cv2 없으면 그대로)."""
    try:
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)
    except Exception:
        return mask


def blur_holes(rgb: np.ndarray, hole_mask: np.ndarray, blur_sigma: float = 12.0) -> np.ndarray:
    """드래그 중 바깥(빈) 영역만 뿌옇게 — 중앙(피사체) 선명도는 유지."""
    if not hole_mask.any():
        return rgb
    out = rgb.copy()
    try:
        import cv2
        soft = cv2.GaussianBlur(rgb, (0, 0), blur_sigma)
        out[hole_mask] = soft[hole_mask]
        return out
    except Exception:
        k, pad = 11, 5
        padded = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), mode="edge").astype(np.float32)
        acc = np.zeros_like(out, dtype=np.float32)
        for dy in range(k):
            for dx in range(k):
                acc += padded[dy:dy + out.shape[0], dx:dx + out.shape[1]]
        out[hole_mask] = (acc / (k * k)).astype(np.uint8)[hole_mask]
        return out
