"""
reframe_yaw.py — yaw×pitch gsplat 프리렌더 그리드

슬라이더 수치 × angle_step = 실제 각도 (0 → 0°, 16 → 48°, -5 → -15°).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import sharp_render
from task_nvs_sharp import SharpScene


@dataclass
class ViewGrid:
    yaws: np.ndarray
    pitches: np.ndarray
    images: List[np.ndarray]
    alphas: List[np.ndarray]

    def nearest(self, yaw: float, pitch: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        d = (self.yaws - float(yaw)) ** 2 + (self.pitches - float(pitch)) ** 2
        idx = int(np.argmin(d))
        return self.images[idx], self.alphas[idx]


YawStrip = ViewGrid


def build_view_grid(
    scene: SharpScene,
    *,
    yaw_idx_min: int = -16,
    yaw_idx_max: int = 16,
    pitch_idx_min: int = -5,
    pitch_idx_max: int = 5,
    angle_step: float = 3.0,
    out_long: int = 1024,
    max_disparity: float = 0.08,
    progress=None,
) -> ViewGrid:
    """index 격자 × angle_step 으로 yaw×pitch 프리렌더."""
    device = sharp_render.require_cuda()
    yaw_indices = np.arange(yaw_idx_min, yaw_idx_max + 1, dtype=np.int64)
    pitch_indices = np.arange(pitch_idx_min, pitch_idx_max + 1, dtype=np.int64)
    pairs = [
        (float(yi * angle_step), float(pi * angle_step))
        for pi in pitch_indices
        for yi in yaw_indices
    ]
    n = len(pairs)

    gaussians = sharp_render.scene_to_gaussians(scene)
    width, height = sharp_render.output_resolution(scene, out_long)
    f_px = scene.f_px * (width / scene.width)
    camera_model = sharp_render.setup_camera(gaussians, width, height, f_px, device)
    renderer = sharp_render.get_renderer() if sharp_render.gsplat_cuda_ready() else None
    offset_x, offset_y = sharp_render.camera_offsets_m(
        gaussians, width, height, f_px, max_disparity=max_disparity,
    )

    yaw_list: List[float] = []
    pitch_list: List[float] = []
    images: List[np.ndarray] = []
    alphas: List[np.ndarray] = []

    for i, (yaw, pitch) in enumerate(pairs):
        rgb, alpha = sharp_render.render_view(
            scene, yaw, pitch,
            out_long=out_long,
            max_disparity=max_disparity,
            offset_x_m=offset_x,
            offset_y_m=offset_y,
            gaussians=gaussians,
            camera_model=camera_model,
            renderer=renderer,
            device=device,
        )
        yaw_list.append(yaw)
        pitch_list.append(pitch)
        images.append(rgb)
        alphas.append(alpha)
        if progress is not None:
            progress(0.2 + 0.75 * (i + 1) / n, desc=f"Reframe 렌더 {i + 1}/{n}")

    return ViewGrid(
        yaws=np.array(yaw_list, dtype=np.float64),
        pitches=np.array(pitch_list, dtype=np.float64),
        images=images,
        alphas=alphas,
    )
