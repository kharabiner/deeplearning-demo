"""
features/expand.py — Expand (프레임 확장 / 아웃페인팅)

iOS 27 "Expand" 재현: 사진 프레임을 축소해 바깥쪽 영역을 노출시키고,
그 바깥 영역을 SD2 아웃페인팅으로 자연스럽게 생성한다.

드래그 중에는 LDI(또는 depth) 워핑으로 바깥을 블러 미리보기,
[완료] 시 바깥 영역만 SD2 로 채운다.

UI 흐름:
  expand_analyze → (슬라이더) expand_view → expand_commit
"""

from __future__ import annotations

import gradio as gr
import numpy as np

from common import pil_to_numpy, numpy_to_pil, resize_if_needed
import reframe_core as core
import reframe_ldi as rldi
from .shared import (
    DEVICE, PREVIEW_MAX, FALLBACK_PARALLAX, HIDDEN, VISIBLE,
    depth_disp, inpaint_commit,
)


# ── 깊이 워핑 렌더 (LDI 우선, 실패 시 단순 depth) ───────────────────────────────
def _render_depth(img_np, disp, plate, move, smooth, frame_zoom=1.0):
    if plate is not None:
        return rldi.render_ldi(
            plate, move, z_near=1.0, z_far=FALLBACK_PARALLAX,
            frame_zoom=float(frame_zoom), smooth=smooth, device=DEVICE,
        )
    if smooth:
        out, inner, outer = core.warp_image(
            img_np, disp, move, z_near=1.0, z_far=FALLBACK_PARALLAX,
            smooth=True, frame_zoom=float(frame_zoom), return_outer=True, device=DEVICE,
        )
        return out, outer, inner
    out, inner = core.warp_image(
        img_np, disp, move, z_near=1.0, z_far=FALLBACK_PARALLAX, smooth=False, device=DEVICE,
    )
    return out, np.zeros(inner.shape, dtype=bool), inner


def _expand_view(disp, plate, img_np, extend):
    move = core.CameraMove()
    out, outer, inner = _render_depth(img_np, disp, plate, move, smooth=True,
                                      frame_zoom=float(extend))
    blur = outer if plate is not None else (outer | inner)
    return core.fill_preview(out, blur) if blur.any() else out


# ── 1) 분석 ─────────────────────────────────────────────────────────────────────
def expand_analyze(image, progress=gr.Progress()):
    if image is None:
        raise gr.Error("먼저 왼쪽에 사진을 업로드하세요.")
    small = resize_if_needed(image.convert("RGB"), max_size=PREVIEW_MAX)
    img_np = pil_to_numpy(small)
    progress(0.4, desc="깊이 추정")
    disp = depth_disp(small)
    plate = None
    ldi_msg = "단순 depth 워핑 (LDI 미적용)"
    try:
        plate = rldi.build_plate(small, disp, DEVICE)
        ldi_msg = "LDI 2-레이어 적용 (전경 립 분리)"
    except Exception as e:
        print(f"[expand] LDI build_plate 실패 → depth 폴백: {e}")
        plate = None
    canvas0 = _expand_view(disp, plate, img_np, 1.0)
    return (
        None, disp, plate, img_np, None, "expand",
        gr.update(value=canvas0, visible=True),
        HIDDEN,
        f"Expand · {ldi_msg} · 슬라이더로 프레임 축소 → 바깥 블러 → [완료]로 채움",
        HIDDEN, VISIBLE, HIDDEN,
    )


# ── 2) 미리보기 ─────────────────────────────────────────────────────────────────
def expand_view(disp, plate, img_np, extend):
    if img_np is None:
        return None
    return _expand_view(disp, plate, img_np, extend)


# ── 3) 확정 (아웃페인팅) ────────────────────────────────────────────────────────
def expand_commit(disp, plate, img_np, extend, progress=gr.Progress()):
    if img_np is None:
        raise gr.Error("먼저 Expand를 실행하세요.")
    move = core.CameraMove()
    progress(0.2, desc="프레임 확장")
    out, outer, inner = _render_depth(img_np, disp, plate, move, smooth=True,
                                      frame_zoom=float(extend))
    fill = core.dilate_mask(outer | inner, iterations=3)
    return inpaint_commit(numpy_to_pil(out), fill, progress, desc="아웃페인팅")
