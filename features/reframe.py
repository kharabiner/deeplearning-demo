"""
features/reframe.py — Reframe · yaw/pitch · gsplat · 프리렌더 그리드

  슬라이더: 정수 수치 (좌우 -16~+16, 상하 -5~+5) — 각도 아님
  실제 회전: 수치 × 3°  (예: 16 → 48°, -5 → -15°)
  [완료]: 동일 gsplat 정확 각도 + 바깥 DreamShaper
"""

from __future__ import annotations

import gradio as gr
import numpy as np

from common import pil_to_numpy, numpy_to_pil, resize_if_needed, free_memory
import inpaint as rinp
import sharp_render
import reframe_yaw
import task_nvs_sharp
from .shared import (
    DEVICE, PREVIEW_MAX, COMMIT_LONG,
    HIDDEN, VISIBLE,
    vlm_caption_reframe, feather_composite, dilate_mask, blur_holes,
)

ANGLE_STEP = 3.0

# 슬라이더 = 정수 수치 (step 1). 실제 각도(°) = 수치 × ANGLE_STEP
YAW_IDX_MIN = -16
YAW_IDX_MAX = 16
PITCH_IDX_MIN = -5
PITCH_IDX_MAX = 5

# SHARP eye 매핑: 슬라이더 최대 수치 × ANGLE_STEP
YAW_RANGE_DEG = float(YAW_IDX_MAX) * ANGLE_STEP    # 16 → 48°
PITCH_RANGE_DEG = float(PITCH_IDX_MAX) * ANGLE_STEP  # 5 → 15°

MAX_DISPARITY = 0.10
OUTER_BLUR_SIGMA = 8.0          # 미리보기용 (드래그)
INPAINT_BLUR_SIGMA = 4.0        # SD 입력: 가벼운 맥락만
ALPHA_HOLE_THRESH = 0.02
ALPHA_SOFT_THRESH = 0.35        # gsplat 반투명 fringe → SD 마스크 포함
FILL_DILATE_ITER = 6
FILL_TELEA_RATIO = 0.92         # 거의 전체가 구멍일 때만 telea
SD15_STEPS = 30
SD15_GUIDANCE = 8.0
SD15_INPAINT_LONG = 768
REFRAME_SD15_NEGATIVE = (
    "new object, duplicate, extra limbs, "
    "blurry, distorted, artifacts, watermark, text, low quality, deformed, "
    "visible seam, harsh edge, smudge"
)

# ui 호환 (슬라이더 step=1 index)
YAW_STEP = 1.0
PITCH_STEP = 1.0
YAW_MIN = YAW_IDX_MIN
YAW_MAX = YAW_IDX_MAX
PITCH_MIN = PITCH_IDX_MIN
PITCH_MAX = PITCH_IDX_MAX


def idx_to_deg(idx: float) -> float:
    """슬라이더 수치 → 실제 회전 각도(°)."""
    return float(idx) * ANGLE_STEP


def idx_to_yaw_deg(idx: float) -> float:
    return idx_to_deg(idx)


def idx_to_pitch_deg(idx: float) -> float:
    return idx_to_deg(idx)


def _at_rest(yaw_idx: float, pitch_idx: float) -> bool:
    return int(round(float(yaw_idx))) == 0 and int(round(float(pitch_idx))) == 0


def _preview_holes(alpha: np.ndarray) -> np.ndarray:
    """테두리와 연결된 미커버(저알파)만 — 내부는 선명 유지."""
    hole = alpha < ALPHA_HOLE_THRESH
    if not hole.any():
        return hole
    try:
        import cv2
    except Exception:
        return hole
    h = hole.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(h, connectivity=8)
    H, W = alpha.shape
    border = np.zeros((H, W), dtype=bool)
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        ht = int(stats[i, cv2.CC_STAT_HEIGHT])
        if x == 0 or y == 0 or x + w >= W or y + ht >= H:
            border[labels == i] = True
    return border


def _outer_fill_mask(alpha: np.ndarray) -> np.ndarray:
    """테두리 구멍 + gsplat 반투명 fringe → SD가 채울 영역."""
    core = _preview_holes(alpha)
    if not core.any():
        return core
    try:
        import cv2
        k = np.ones((5, 5), np.uint8)
        grown = cv2.dilate(core.astype(np.uint8), k, iterations=4)
        fringe = (alpha < ALPHA_SOFT_THRESH) & (grown > 0)
        return fringe | core
    except Exception:
        return core


def _inpaint_canvas(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """SD 입력: 선명 gsplat + 구멍만 가벼운 블러(맥락). 미리보기용 강블러와 분리."""
    holes = _outer_fill_mask(alpha)
    if not holes.any():
        return rgb
    return blur_holes(rgb, holes, blur_sigma=INPAINT_BLUR_SIGMA)


def _present(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """미리보기: 테두리 구멍만 강블러."""
    holes = _preview_holes(alpha)
    return blur_holes(rgb, holes, blur_sigma=OUTER_BLUR_SIGMA) if holes.any() else rgb


def reframe_analyze(image, progress=gr.Progress()):
    if image is None:
        raise gr.Error("먼저 왼쪽에 사진을 업로드하세요.")

    rinp.unload_expand_sd15()
    from features.clean_up import unload_sam
    unload_sam()
    free_memory(DEVICE)

    try:
        sharp_render.require_cuda()
    except RuntimeError as e:
        raise gr.Error(str(e))

    backend = sharp_render.renderer_label()
    progress(0.05, desc="Reframe — 준비 중...")
    orig = image.convert("RGB")
    small = resize_if_needed(orig, max_size=PREVIEW_MAX)
    img_np = pil_to_numpy(small)

    progress(0.12, desc="Reframe — SHARP 3D 분석")
    scene = task_nvs_sharp.predict(orig, device=DEVICE)
    label = f"SHARP ({scene.num_gaussians:,} gaussians) · {backend}"

    progress(0.18, desc="Reframe — yaw×pitch 그리드 렌더 (gsplat)")
    grid = reframe_yaw.build_view_grid(
        scene,
        yaw_idx_min=YAW_IDX_MIN, yaw_idx_max=YAW_IDX_MAX,
        pitch_idx_min=PITCH_IDX_MIN, pitch_idx_max=PITCH_IDX_MAX,
        angle_step=ANGLE_STEP,
        yaw_range_deg=YAW_RANGE_DEG,
        pitch_range_deg=PITCH_RANGE_DEG,
        out_long=PREVIEW_MAX, max_disparity=MAX_DISPARITY,
        progress=progress,
    )

    task_nvs_sharp.unload_predictor()
    scene = scene.to("cpu")
    free_memory(DEVICE)

    ny = YAW_IDX_MAX - YAW_IDX_MIN + 1
    np_ = PITCH_IDX_MAX - PITCH_IDX_MIN + 1
    n = len(grid.yaws)
    status = (
        f"✅ Reframe 준비 · {label}\n"
        f"📍 좌우 {YAW_IDX_MIN}~{YAW_IDX_MAX} · 상하 {PITCH_IDX_MIN}~{PITCH_IDX_MAX} "
        f"(×{ANGLE_STEP:g}° → 최대 ±{YAW_RANGE_DEG:g}° / ±{PITCH_RANGE_DEG:g}°) · "
        f"{n}프레임 ({ny}×{np_})\n"
        f"🎚 슬라이더 0=정면 · 수치 1당 {ANGLE_STEP:g}° · [완료] gsplat + DreamShaper"
    )
    return (
        scene, None, grid, img_np, None, "reframe",
        gr.update(value=img_np, visible=True),
        HIDDEN,
        status,
        VISIBLE, HIDDEN, HIDDEN,
        gr.update(value=0), gr.update(value=0),
    )


def reframe_view(_scene, _disp, grid, img_np, yaw_idx, pitch_idx):
    if img_np is None:
        return None
    if _at_rest(yaw_idx, pitch_idx):
        return img_np
    if grid is None:
        return img_np
    yaw_deg = idx_to_yaw_deg(yaw_idx)
    pitch_deg = idx_to_pitch_deg(pitch_idx)
    rgb, alpha = grid.nearest(yaw_deg, pitch_deg)
    return _present(rgb, alpha)


def reframe_commit(scene, _disp, img_np, yaw_idx, pitch_idx, progress=gr.Progress()):
    if img_np is None or scene is None:
        raise gr.Error("먼저 Reframe를 실행하세요.")
    if _at_rest(yaw_idx, pitch_idx):
        return numpy_to_pil(img_np), "각도 변경이 거의 없습니다."

    yaw_deg = idx_to_yaw_deg(yaw_idx)
    pitch_deg = idx_to_pitch_deg(pitch_idx)

    try:
        sharp_render.require_cuda()
    except RuntimeError as e:
        raise gr.Error(str(e))

    progress(0.25, desc="Reframe — gsplat 정확 각도 렌더")
    rgb, alpha = sharp_render.render_view(
        scene, yaw_deg, pitch_deg,
        yaw_max_deg=YAW_RANGE_DEG,
        pitch_max_deg=PITCH_RANGE_DEG,
        out_long=COMMIT_LONG,
        max_disparity=MAX_DISPARITY,
    )
    preview = _present(rgb, alpha)
    free_memory(DEVICE)
    preview_pil = numpy_to_pil(preview)
    sharp_pil = numpy_to_pil(rgb)

    fill = dilate_mask(_outer_fill_mask(alpha), iterations=FILL_DILATE_ITER)
    fill_ratio = float(fill.sum()) / float(fill.size)
    tag = f"yaw={yaw_deg:g}° pitch={pitch_deg:g}°"
    if int(fill.sum()) < 30:
        return preview_pil, f"✅ Reframe · {tag} · gsplat · 채울 영역 없음"

    import inpaint as rinp

    if fill_ratio > FILL_TELEA_RATIO:
        inp = rinp.get_inpainter("opencv", DEVICE)
        return inp.inpaint(preview_pil, fill), (
            f"✅ Reframe · {tag} · gsplat + telea (빈 {fill_ratio:.0%})"
        )

    canvas = _inpaint_canvas(rgb, alpha)
    canvas_pil = numpy_to_pil(canvas)

    progress(0.55, desc="Reframe — DreamShaper 바깥 생성")
    prompt = vlm_caption_reframe(canvas_pil)
    try:
        inp = rinp.get_inpainter("sd15", DEVICE)
        result = inp.inpaint(
            canvas_pil, fill, prompt=prompt,
            steps=SD15_STEPS, guidance=SD15_GUIDANCE,
            negative_prompt=REFRAME_SD15_NEGATIVE,
            long=SD15_INPAINT_LONG,
        )
        inp.unload()
    except Exception as e:
        raise gr.Error(f"인페인팅 실패: {e}")

    result = feather_composite(sharp_pil, result, fill, feather=3.0)
    return result, f"✅ Reframe · {tag} · gsplat + DreamShaper · fill={fill_ratio:.1%}"
