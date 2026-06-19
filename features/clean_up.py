"""
features/clean_up.py — Clean Up (객체 지우기)

iOS 27 "Clean Up" 재현: 사진에서 지우고 싶은 사물/사람을 브러시로 문질러
선택하면 SAM2 로 자동 세그멘테이션 후 DreamShaper 인페인팅으로 깔끔히 제거한다.
(텍스트 검색 → Grounding DINO 탐지 → SAM2 경로도 지원)

UI 흐름:
  clean_up_prepare → (브러시) clean_up_brush / (텍스트) clean_up_detect
  → (초기화) clean_up_clear → clean_up_commit
"""

from __future__ import annotations

import gradio as gr
import numpy as np
from PIL import Image

from common import free_memory, pil_to_numpy, numpy_to_pil, resize_if_needed
import inpaint as rinp
from .shared import (
    DEVICE, PREVIEW_MAX, HIDDEN, VISIBLE,
    vlm_caption, feather_composite, dilate_mask,
)


# Clean Up 세션 동안 SAM2 GPU 상주 (브러시마다 reload 방지)
_sam_cache = None


def unload_sam():
    """SAM2 세션 종료 — [완료] 또는 다른 탭 진입 시."""
    global _sam_cache
    if _sam_cache is not None:
        del _sam_cache
        _sam_cache = None
        free_memory(DEVICE)


def _ensure_sam(progress=None):
    global _sam_cache
    if _sam_cache is None:
        if progress is not None:
            progress(0.05, desc="SAM2 로드")
        import task_segmentation_sam2 as seg
        _sam_cache = seg.load_model(DEVICE)
    return _sam_cache


# ── 마스크/에디터 헬퍼 ──────────────────────────────────────────────────────────
def _pick_object_mask(results, H, W):
    """SAM2 후보 중 '객체 전체'에 가장 가까운 마스크 선택."""
    total = float(H * W)
    scored = [(r, float(r["mask"].sum())) for r in results]
    good = [
        (r, area) for r, area in scored
        if 0.0008 * total <= area <= 0.92 * total and r.get("score", 0.0) >= 0.5
    ]
    pool = good or scored
    return max(pool, key=lambda ra: ra[1])[0]["mask"]


def _overlay(img_np, mask):
    """선택 마스크를 원본 위에 색으로 겹친 미리보기."""
    import task_segmentation_sam2 as seg
    pil = numpy_to_pil(img_np)
    if mask is None or not mask.any():
        return pil
    return numpy_to_pil(seg.masks_to_overlay(pil, [{"mask": mask}]))


def _editor_value(img_np, mask=None):
    """ImageEditor용 dict — 배경에 마스크 오버레이, 브러시 레이어 초기화."""
    bg = _overlay(img_np, mask)
    arr = pil_to_numpy(bg) if isinstance(bg, Image.Image) else bg
    return {"background": arr, "layers": [], "composite": None}


def _as_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Gradio ImageEditor numpy (uint8 또는 0~1 float) → uint8 HWC."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if np.issubdtype(arr.dtype, np.floating) and float(arr.max()) <= 1.0 + 1e-3:
        arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8)


def _layer_stroke(layer: np.ndarray, H: int, W: int) -> np.ndarray:
    """단일 브러시 레이어 → bool (H,W)."""
    arr = _as_uint8_rgb(layer)
    if arr.shape[0] != H or arr.shape[1] != W:
        arr = np.asarray(Image.fromarray(arr).resize((W, H), Image.NEAREST))
    if arr.shape[2] >= 4 and np.any(arr[..., 3] > 20):
        return arr[..., 3] > 20
    return np.any(arr[..., :3] > 30, axis=-1)


def _extract_stroke_mask(editor_value, H: int, W: int) -> np.ndarray:
    """ImageEditor brush layer → bool (H,W) stroke mask."""
    if editor_value is None or isinstance(editor_value, Image.Image):
        return np.zeros((H, W), dtype=bool)
    if not isinstance(editor_value, dict):
        return np.zeros((H, W), dtype=bool)

    # 방법 1: composite 와 background 비교 (브러시로 칠한 차이 감지)
    composite = editor_value.get("composite")
    background = editor_value.get("background")
    if composite is not None and background is not None:
        comp_arr = _as_uint8_rgb(composite)
        bg_arr = _as_uint8_rgb(background)
        if comp_arr.shape[:2] != (H, W):
            comp_arr = np.asarray(Image.fromarray(comp_arr).resize((W, H), Image.LANCZOS))
        if bg_arr.shape[:2] != (H, W):
            bg_arr = np.asarray(Image.fromarray(bg_arr).resize((W, H), Image.LANCZOS))
        if comp_arr.shape == bg_arr.shape:
            diff = np.abs(comp_arr.astype(np.float32) - bg_arr.astype(np.float32))
            if diff.ndim == 3:
                diff = np.max(diff, axis=2)
            stroke = diff > 10
            if stroke.sum() > 0:
                return stroke

    # 방법 2: layers 알파/색 채널
    stroke = np.zeros((H, W), dtype=bool)
    for layer in editor_value.get("layers") or []:
        if layer is None:
            continue
        stroke |= _layer_stroke(layer, H, W)
    return stroke


def _sam_mask_from_stroke(img_np, stroke: np.ndarray, progress=None):
    """브러시 획 → SAM2 객체 마스크 (실패 시 None)."""
    import task_segmentation_sam2 as seg

    H, W = img_np.shape[:2]
    pil = numpy_to_pil(img_np)
    proc, model = _ensure_sam(progress)
    results = seg.run_with_stroke_mask(pil, proc, model, stroke, device=DEVICE)
    if not results:
        return None
    return _pick_object_mask(results, H, W)


# ── 1) 준비 ─────────────────────────────────────────────────────────────────────
def clean_up_prepare(image, progress=gr.Progress()):
    if image is None:
        raise gr.Error("먼저 왼쪽에 사진을 업로드하세요.")
    rinp.unload_expand_sd15()
    unload_sam()
    free_memory(DEVICE)
    small = resize_if_needed(image.convert("RGB"), max_size=PREVIEW_MAX)
    img_np = pil_to_numpy(small)
    progress(0.2, desc="SAM2 준비")
    _ensure_sam(progress)
    return (
        None, None, None, img_np, None, "clean_up",
        HIDDEN,
        gr.update(value=_editor_value(img_np), visible=True),
        "Clean Up: 지우고 싶은 객체를 **브러시로 문질러서 선택**하세요. (텍스트 검색도 가능)",
        HIDDEN, HIDDEN, VISIBLE,
        gr.skip(), gr.skip(),
    )


# ── 2a) 브러시 선택 → SAM2 ──────────────────────────────────────────────────────
def clean_up_brush(mode, img_np, mask, editor_value, progress=gr.Progress()):
    if mode != "clean_up":
        return gr.skip(), gr.skip(), gr.skip()
    if img_np is None:
        raise gr.Error("먼저 Clean Up 버튼을 누르세요.")

    H, W = img_np.shape[:2]
    stroke = _extract_stroke_mask(editor_value, H, W)
    if not stroke.any():
        # 에디터 갱신 echo / blob 미완 — mask 유지
        return gr.skip(), gr.skip(), gr.skip()

    progress(0.1, desc="SAM2 선택")
    new_mask = _sam_mask_from_stroke(img_np, stroke, progress)
    if new_mask is None:
        return _editor_value(img_np, mask), mask, "선택 실패 — 다른 위치를 문질러 보세요."

    if mask is not None and mask.any():
        new_mask = mask | new_mask
    n_px = int(new_mask.sum())
    return (
        _editor_value(img_np, new_mask),
        new_mask,
        f"선택됨 · {n_px:,} px · 추가 문지르기 가능 · [완료]로 지우기",
    )


# ── 2b) 텍스트 검색 → Grounding DINO + SAM2 ─────────────────────────────────────
def clean_up_detect(img_np, text_prompt, progress=gr.Progress()):
    if img_np is None:
        raise gr.Error("먼저 Clean Up 버튼을 누르세요.")
    prompt = (text_prompt or "").strip()
    if not prompt:
        raise gr.Error("검색할 객체를 영어로 입력하세요. 예: person . bottle .")

    import task_detection_groundingdino as det
    import task_segmentation_sam2 as seg

    pil = numpy_to_pil(img_np)
    progress(0.3, desc="Grounding DINO 탐지")
    dproc, dmodel = det.load_model(DEVICE)
    try:
        result = det.run(pil, dproc, dmodel, text_prompt=prompt, device=DEVICE)
    finally:
        del dproc, dmodel
        free_memory(DEVICE)

    boxes = result["boxes"].tolist()
    if not boxes:
        return _editor_value(img_np, None), None, f"탐지 없음 · prompt={prompt}"

    progress(0.55, desc="SAM2 세그멘테이션")
    sproc, smodel = _ensure_sam(progress)
    seg_results = seg.run_with_boxes(pil, sproc, smodel, boxes, device=DEVICE)

    combined = np.zeros(img_np.shape[:2], dtype=bool)
    for r in seg_results:
        combined |= r["mask"]
    labels = ", ".join(result["labels"][:4])
    n_px = int(combined.sum())
    return (
        _editor_value(img_np, combined), combined,
        f"탐지 {len(boxes)}개 · {labels} · {n_px:,} px",
    )


# ── 3) 선택 초기화 ──────────────────────────────────────────────────────────────
def clean_up_clear(img_np):
    if img_np is None:
        return None, None, "선택 초기화됨"
    return _editor_value(img_np), None, "선택 초기화 — 다시 문지르세요."


# ── 4) 확정 (제거) ──────────────────────────────────────────────────────────────
def clean_up_commit(img_np, mask, editor_value, progress=gr.Progress()):
    if img_np is None:
        return gr.skip(), gr.skip(), gr.skip(), "이미지가 없습니다. Clean Up을 다시 시작하세요."

    H, W = img_np.shape[:2]
    if mask is None or not mask.any():
        stroke = _extract_stroke_mask(editor_value, H, W)
        if stroke.any():
            progress(0.15, desc="Clean Up — SAM2 선택")
            mask = _sam_mask_from_stroke(img_np, stroke, progress)
    if mask is None or not mask.any():
        raise gr.Error("먼저 지울 객체를 브러시로 문질러 선택하세요.")

    unload_sam()
    fill = dilate_mask(mask, iterations=6)
    orig_pil = numpy_to_pil(img_np)
    progress(0.35, desc="Clean Up — 객체 제거 중")

    try:
        prompt = vlm_caption(orig_pil)
        inp = rinp.get_inpainter("sd15", DEVICE)
        filled = inp.inpaint(orig_pil, fill, prompt=prompt)
        inp.unload()
    except Exception as e:
        raise gr.Error(f"제거 실패(DreamShaper): {e}")

    result = feather_composite(orig_pil, filled, fill)
    result_np = pil_to_numpy(result)
    return (
        gr.update(value=result, visible=True),
        HIDDEN,
        result_np,
        f"완료 · Clean Up · DreamShaper · {prompt[:40]} · 우상단 아이콘으로 다운로드",
    )
