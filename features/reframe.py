"""
features/reframe.py — Reframe (시점 변경) · SHARP 단일 루트

iOS 27 "Spatial Reframing" 재현: 사진을 찍기 전에 카메라 각도를 바꾸는 것처럼
시점을 회전한다. 가려졌던 면은 SHARP(단일 이미지 → 3D Gaussian)가 생성한다.

단일 루트(미리보기·확정 모두 SHARP):
  버튼 → (로딩) SHARP 3D 분석 + yaw×pitch 격자 사전 렌더(build_grid)
  → 슬라이더 드래그 시 가장 가까운 격자 프레임을 즉시 표시(바깥은 블러)
  → [완료] 시 정확한 각도로 render_exact + 바깥 영역 SD2 생성

depth 워핑/LDI 는 쓰지 않는다(미리보기와 확정이 같은 SHARP 결과라 일치).
큰 각도는 왜곡되므로 소폭 시점 이동(±12° 이내) 권장.

상태 사용:
  st_scene = SharpScene, st_plate = ReframeGrid, st_img = 원본 미리보기 np
  (st_disp 슬롯은 Reframe 에서 사용 안 함)

UI 흐름:
  reframe_analyze → (슬라이더) reframe_view → reframe_commit
"""

from __future__ import annotations

import gradio as gr
import numpy as np

from common import pil_to_numpy, numpy_to_pil, resize_if_needed, free_memory
import reframe_core as core
import reframe_sharp
import task_nvs_sharp
from .shared import (
    DEVICE, PREVIEW_MAX, COMMIT_LONG,
    HIDDEN, VISIBLE,
    vlm_caption_reframe, feather_composite,
)

# 격자 사전 렌더 설정 (분석 시 1회) — 슬라이더 범위와 맞춤
GRID_YAW_MAX = 16.0
GRID_PITCH_MAX = 10.0
GRID_N_YAW = 11           # yaw 격자 (≈3.2° 간격)
GRID_N_PITCH = 5          # pitch 격자 (5° 간격)
GRID_LONG = 768           # 격자 미리보기 해상도(긴 변)


def _scene_pivot_z(scene) -> float:
    """배경/확정 렌더가 같은 pivot 을 쓰도록 장면 median z 사용."""
    import torch
    return float(torch.median(scene.means[:, 2]).item())


def _blur_outer(rgb: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """바깥(테두리 미커버) 영역만 블러 — iOS 드래그 프리뷰 방식."""
    holes = core.preview_holes(~cov)
    return core.fill_preview(rgb, holes) if holes.any() else rgb


# ── 1) 분석 (로딩) ──────────────────────────────────────────────────────────────
def reframe_analyze(image, progress=gr.Progress()):
    if image is None:
        raise gr.Error("먼저 왼쪽에 사진을 업로드하세요.")

    progress(0.05, desc="Reframe — 준비 중...")
    orig = image.convert("RGB")
    small = resize_if_needed(orig, max_size=PREVIEW_MAX)
    img_np = pil_to_numpy(small)

    progress(0.15, desc="Reframe — 사진을 3D로 분석 중 (SHARP)")
    scene = task_nvs_sharp.predict(orig, device=DEVICE)
    mode_label = f"SHARP ({scene.num_gaussians:,} gaussians)"

    # yaw×pitch 격자 사전 렌더 → 드래그 시 즉시 표시(실시간 SHARP 미리보기)
    grid = reframe_sharp.build_grid(
        scene,
        yaw_max=GRID_YAW_MAX, pitch_max=GRID_PITCH_MAX,
        n_yaw=GRID_N_YAW, n_pitch=GRID_N_PITCH,
        out_long=GRID_LONG, close_holes=True, supersample=1,
        device=DEVICE, progress=progress,
    )

    # VRAM 회수: SHARP predictor 언로드 + scene 은 CPU 로 내려 저장
    # (commit 때 VLM·SD 인페인팅이 GPU 에 온전히 올라가도록 — 8GB 대응)
    task_nvs_sharp.unload_predictor()
    scene = scene.to("cpu")
    free_memory(DEVICE)

    # 0°에서는 원본을 그대로(가장 선명) 보여줌
    canvas0 = img_np
    status = (
        f"✅ Reframe 준비 완료 · {mode_label}\n"
        f"📍 슬라이더로 각도 조정 → **실시간 미리보기** (SHARP 격자, 바깥은 블러)\n"
        f"🎬 [완료] 버튼 → 정확한 각도로 SHARP 고품질 렌더 + SD2로 바깥 영역 생성"
    )
    return (
        scene, None, grid, img_np, None, "reframe",
        gr.update(value=canvas0, visible=True),
        HIDDEN,
        status,
        VISIBLE, HIDDEN, HIDDEN,
    )


# ── 2) 미리보기 (격자 최근접) ───────────────────────────────────────────────────
def reframe_view(_scene, _disp, grid, img_np, yaw, pitch):
    if img_np is None:
        return None
    if abs(float(yaw)) < 0.5 and abs(float(pitch)) < 0.5:
        return img_np
    if grid is None:
        return img_np
    rgb, cov = grid.nearest(float(yaw), float(pitch))
    return _blur_outer(rgb, cov)


# ── 3) 확정 (정확 각도 SHARP 렌더 + SD2 생성) ───────────────────────────────────
def reframe_commit(scene, _disp, img_np, yaw, pitch, progress=gr.Progress()):
    if img_np is None or scene is None:
        raise gr.Error("먼저 Reframe를 실행하세요.")

    if abs(float(yaw)) < 0.5 and abs(float(pitch)) < 0.5:
        return numpy_to_pil(img_np), "각도 변경이 거의 없습니다. 슬라이더를 조정해주세요."

    move = core.CameraMove(yaw_deg=float(yaw), pitch_deg=float(pitch))
    pivot_z = _scene_pivot_z(scene)

    # 미리보기(격자)와 동일한 렌더 파라미터 → 확정 결과가 미리보기와 일치.
    # trim_coverage=False: 테두리 커버리지를 깎지 않아 ~cov(빈 곳) 과대평가 방지.
    progress(0.2, desc="Reframe — SHARP로 정확한 시점 렌더링 중")
    out, cov = reframe_sharp.render_exact(
        scene, move, pivot_z=pivot_z,
        out_long=COMMIT_LONG, supersample=1, trim_coverage=False, device=DEVICE,
    )
    free_memory(DEVICE)   # 렌더 임시 텐서 해제 → VLM·SD 용 VRAM 확보
    out_pil = numpy_to_pil(out)

    # 바깥(테두리 미커버) 영역만 인페인팅 대상
    fill = core.dilate_mask(core.preview_holes(~cov), iterations=2)
    fill_ratio = float(fill.sum()) / float(fill.size)

    # 빈 곳이 거의 없으면 그대로 반환
    if int(fill.sum()) < 30:
        return out_pil, f"✅ Reframe 완료 · yaw={yaw}° pitch={pitch}° · SHARP · 채울 영역 없음"

    import inpaint as rinp

    # 안전장치: 빈 곳이 과도하게 크면(큰 각도/렌더 커버리지 불량) SD2 생성은
    # 헛것을 그릴 수 있으므로, OpenCV telea 로 매끈하게만 채워 점박이를 없앤다.
    if fill_ratio > 0.5:
        print(f"[reframe] fill 영역 과대({fill_ratio:.1%}) → telea 로 매끈 채움(SD2 생략)")
        inp = rinp.get_inpainter("opencv", DEVICE)
        result_pil = inp.inpaint(out_pil, fill)
        return result_pil, (
            f"✅ Reframe 완료 · yaw={yaw}° pitch={pitch}° · SHARP + telea "
            f"(빈 영역 {fill_ratio:.0%} 큼 → 매끈 채움, 각도를 줄이면 SD2 생성)"
        )

    progress(0.5, desc="Reframe — SD2로 바깥 영역 생성 중")
    prompt = vlm_caption_reframe(out_pil)
    try:
        inp = rinp.get_inpainter("sd2", DEVICE)
        result_pil = inp.inpaint(out_pil, fill, prompt=prompt)
        inp.unload()
    except Exception as e:
        raise gr.Error(f"인페인팅 실패(SD2): {e}")

    progress(0.9, desc="Reframe — 마무리 중 (블렌딩)")
    result_pil = feather_composite(out_pil, result_pil, fill, feather=3.0)

    return result_pil, (
        f"✅ Reframe 완료 · yaw={yaw}° pitch={pitch}° · SHARP + SD2 · "
        f"prompt={prompt[:40]} · fill={fill_ratio:.1%}"
    )
