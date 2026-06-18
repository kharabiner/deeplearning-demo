"""
features/reframe.py — Reframe (시점 변경)

iOS 27 "Spatial Reframing" 재현: 사진을 찍기 전에 카메라 각도를 바꾸는 것처럼
시점을 회전한다. 가려졌던 면은 SHARP(단일 이미지 → 3D Gaussian)로 생성한다.

요구 동작:
  버튼 → (로딩) SHARP 3D 분석 + 깊이 + LDI 플레이트
  → 슬라이더 드래그 시 **미리보기**에서 사람/사물은 깔끔히 회전, 바깥은 블러
  → [완료] 시 SHARP 고품질 렌더 + 바깥 영역 SD2 생성

미리보기는 reframe_ldi.render_ldi(전경/사람 빌보드 분리)로 늘어남 없이 회전시키고,
바깥(프레임 밖)만 블러 처리한다. 플레이트 생성 실패 시 단순 depth 워핑으로 폴백.

UI 흐름:
  reframe_analyze → (슬라이더) reframe_view → reframe_commit
"""

from __future__ import annotations

import gradio as gr
import numpy as np
from PIL import Image

from common import pil_to_numpy, numpy_to_pil, resize_if_needed
import reframe_core as core
import reframe_ldi as rldi
import reframe_sharp
import task_nvs_sharp
from .shared import (
    DEVICE, PREVIEW_MAX, COMMIT_LONG, RENDER_SS_COMMIT, FALLBACK_PARALLAX,
    HIDDEN, VISIBLE,
    depth_disp, vlm_caption_reframe, feather_composite, resize_mask,
)


# ── 미리보기 (LDI 우선, depth 폴백) ─────────────────────────────────────────────
def _ldi_preview(plate, yaw, pitch):
    """LDI 레이어 렌더: 사람/사물 깔끔히 회전 + 바깥 블러 (iOS 방식)."""
    move = core.CameraMove(yaw_deg=float(yaw), pitch_deg=float(pitch))
    out, outer, residual = rldi.render_ldi(
        plate, move, z_near=1.0, z_far=FALLBACK_PARALLAX, smooth=True, device=DEVICE,
    )
    blur = core.preview_holes(outer | residual)
    return core.fill_preview(out, blur) if blur.any() else out


def _depth_preview(disp, img_np, yaw, pitch):
    """폴백 미리보기: 원본 픽셀 depth warp + 테두리만 블러."""
    move = core.CameraMove(yaw_deg=float(yaw), pitch_deg=float(pitch))
    warped, _hole, outer = core.warp_image(
        img_np, disp, move, z_near=1.0, z_far=FALLBACK_PARALLAX,
        smooth=True, return_outer=True, device=DEVICE,
    )
    blur = core.preview_holes(outer)
    return core.fill_preview(warped, blur) if blur.any() else warped


def _preview(disp, plate, img_np, yaw, pitch):
    if abs(float(yaw)) < 0.5 and abs(float(pitch)) < 0.5:
        return img_np
    if plate is not None:
        try:
            return _ldi_preview(plate, yaw, pitch)
        except Exception as e:
            print(f"[reframe] LDI 미리보기 실패 → depth 폴백: {e}")
    return _depth_preview(disp, img_np, yaw, pitch)


# ── 확정용 헬퍼 ─────────────────────────────────────────────────────────────────
def _scene_pivot_z(scene, disp):
    """SHARP pivot (render_exact용)."""
    if scene is not None:
        import torch
        return float(torch.median(scene.means[:, 2]).item())
    return float(np.median(core.disparity_to_z_np(disp, 1.0, FALLBACK_PARALLAX)))


def _warp_holes(disp, img_np, yaw, pitch):
    """확정용 hole mask — 프리뷰와 동일 규칙(preview_holes)."""
    move = core.CameraMove(yaw_deg=float(yaw), pitch_deg=float(pitch))
    _warped, hole, outer = core.warp_image(
        img_np, disp, move, z_near=1.0, z_far=FALLBACK_PARALLAX,
        smooth=True, return_outer=True, device=DEVICE,
    )
    return core.preview_holes(outer | hole)


def _fill_mask(disp, img_np, yaw, pitch, out_hw=None):
    """테두리 디오클루전만 인페인팅 (SHARP coverage 전체 제외)."""
    fill = _warp_holes(disp, img_np, yaw, pitch)
    if out_hw is not None:
        fill = resize_mask(fill, out_hw[0], out_hw[1])
    return core.dilate_mask(fill, iterations=2)


# ── 1) 분석 (로딩) ──────────────────────────────────────────────────────────────
def reframe_analyze(image, progress=gr.Progress()):
    if image is None:
        raise gr.Error("먼저 왼쪽에 사진을 업로드하세요.")

    progress(0.1, desc="Reframe — 준비 중...")
    orig = image.convert("RGB")
    small = resize_if_needed(orig, max_size=PREVIEW_MAX)
    img_np = pil_to_numpy(small)

    progress(0.3, desc="Reframe — 사진을 3D로 분석 중 (SHARP)")
    scene = task_nvs_sharp.predict(orig, device=DEVICE)
    mode_label = f"SHARP ({scene.num_gaussians:,} gaussians)"

    progress(0.6, desc="Reframe — 깊이 맵 생성 (미리보기용)")
    disp = depth_disp(small)

    # 미리보기용 LDI 플레이트(전경/사람 분리) → 사람·사물이 깔끔히 회전
    progress(0.8, desc="Reframe — 전경/사람 레이어 분리 (LDI)")
    plate = None
    ldi_msg = "depth 미리보기"
    try:
        plate = rldi.build_plate(small, disp, DEVICE)
        ldi_msg = "LDI 미리보기 (사람/사물 분리)"
    except Exception as e:
        print(f"[reframe] LDI build_plate 실패 → depth 미리보기 폴백: {e}")
        plate = None

    canvas0 = _preview(disp, plate, img_np, 0.0, 0.0)
    status = (
        f"✅ Reframe 준비 완료 · {mode_label} · {ldi_msg}\n"
        f"📍 슬라이더로 각도 조정 → **실시간 미리보기** (사람/사물은 선명, 바깥은 블러)\n"
        f"🎬 [완료] 버튼 → SHARP로 고품질 렌더 + SD2로 바깥 영역 생성"
    )
    return (
        scene, disp, plate, img_np, None, "reframe",
        gr.update(value=canvas0, visible=True),
        HIDDEN,
        status,
        VISIBLE, HIDDEN, HIDDEN,
    )


# ── 2) 미리보기 ─────────────────────────────────────────────────────────────────
def reframe_view(_scene, disp, plate, img_np, yaw, pitch):
    if img_np is None:
        return None
    return _preview(disp, plate, img_np, yaw, pitch)


# ── 3) 확정 (SHARP 렌더 + SD2 생성) ─────────────────────────────────────────────
def reframe_commit(scene, disp, img_np, yaw, pitch, progress=gr.Progress()):
    if img_np is None:
        raise gr.Error("먼저 Reframe를 실행하세요.")

    if abs(float(yaw)) < 0.5 and abs(float(pitch)) < 0.5:
        return numpy_to_pil(img_np), "각도 변경이 거의 없습니다. 슬라이더를 조정해주세요."

    move = core.CameraMove(yaw_deg=float(yaw), pitch_deg=float(pitch))

    # SHARP 전용 고품질 렌더링
    pivot_z = _scene_pivot_z(scene, disp)
    progress(0.2, desc="Reframe — SHARP로 시점 렌더링 중")
    out, cov = reframe_sharp.render_exact(
        scene, move, pivot_z=pivot_z,
        out_long=COMMIT_LONG, supersample=RENDER_SS_COMMIT, device=DEVICE,
    )

    H, W = out.shape[:2]
    img_c, disp_c = img_np, disp
    if img_np.shape[:2] != (H, W):
        img_c = pil_to_numpy(numpy_to_pil(img_np).resize((W, H), Image.LANCZOS))
        try:
            import cv2
            disp_c = cv2.resize(disp.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        except Exception:
            disp_c = disp

    progress(0.4, desc="Reframe — 원본과 렌더 결과 합성 중")
    out = core.composite_photo_anchor(
        img_c, disp_c, yaw, pitch, out, cov,
        z_far=FALLBACK_PARALLAX, device=DEVICE,
    )
    fill = _fill_mask(disp_c, img_c, yaw, pitch, out_hw=out.shape[:2])

    fill_ratio = float(fill.sum()) / float(fill.size)
    if fill_ratio > 0.18:
        print(f"[reframe] fill mask large ({fill_ratio:.1%}) — 테두리만 유지")
        fill = core.preview_holes(fill)

    progress(0.5, desc="Reframe — SD2로 바깥 영역 생성 중")
    prompt = vlm_caption_reframe(numpy_to_pil(out))

    import inpaint as rinp
    try:
        inp = rinp.get_inpainter("sd2", DEVICE)
        result_pil = inp.inpaint(numpy_to_pil(out), fill, prompt=prompt)
        inp.unload()
    except Exception as e:
        raise gr.Error(f"인페인팅 실패(SD2): {e}")

    if int(fill.sum()) >= 30:
        progress(0.9, desc="Reframe — 마무리 중 (블렌딩)")
        result_pil = feather_composite(numpy_to_pil(out), result_pil, fill, feather=3.0)

    return result_pil, f"✅ Reframe 완료 · yaw={yaw}° pitch={pitch}° · SD2 · prompt={prompt[:40]} · fill={fill_ratio:.1%}"
