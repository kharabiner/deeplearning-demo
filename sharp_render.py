"""
sharp_render.py — Apple SHARP + gsplat(CUDA) / PyTorch splat 폴백

gsplat CUDA 사용 가능 시 공식 GSplatRenderer, 아니면 splat_torch 폴백.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from task_nvs_sharp import SharpScene

_RENDERER = None


def gsplat_cuda_ready() -> bool:
    """gsplat CUDA 확장이 로드·컴파일되어 사용 가능한지."""
    try:
        from gsplat.cuda._backend import _C

        return _C is not None
    except Exception:
        return False


def renderer_label() -> str:
    return "gsplat" if gsplat_cuda_ready() else "torch splat"


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Reframe 3D 렌더는 CUDA GPU가 필요합니다."
        )
    return torch.device("cuda")


def scene_to_gaussians(scene: SharpScene):
    """SharpScene → ml-sharp Gaussians3D (batch=1)."""
    from sharp.utils.gaussians import Gaussians3D

    return Gaussians3D(
        mean_vectors=scene.means.unsqueeze(0),
        singular_values=scene.scales.unsqueeze(0),
        quaternions=scene.quats.unsqueeze(0),
        colors=scene.colors.unsqueeze(0),
        opacities=scene.opacities.unsqueeze(0),
    )


def output_resolution(scene: SharpScene, out_long: int) -> Tuple[int, int]:
    """(width, height) — SHARP resolution_px 규약."""
    if scene.width >= scene.height:
        w = out_long
        h = max(1, round(out_long * scene.height / scene.width))
    else:
        h = out_long
        w = max(1, round(out_long * scene.width / scene.height))
    return w, h


def get_renderer():
    """GSplatRenderer 싱글톤 (gsplat CUDA 필요)."""
    global _RENDERER
    if not gsplat_cuda_ready():
        raise RuntimeError(
            "gsplat CUDA를 사용할 수 없습니다. "
            "Windows에서는 Python 3.10 + gsplat 사전 빌드 휠 또는 CUDA Toolkit+VS 빌드가 필요합니다. "
            "현재는 torch splat 폴백이 자동 사용됩니다."
        )
    if _RENDERER is None:
        try:
            from sharp.utils import gsplat as sharp_gsplat
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", str(e))
            raise RuntimeError(
                f"Reframe gsplat 렌더 의존성 누락 ({missing}). "
                "설치: pip install gsplat \"imageio[ffmpeg]\""
            ) from e

        _RENDERER = sharp_gsplat.GSplatRenderer(color_space="linearRGB")
        _RENDERER.eval()
    return _RENDERER


def camera_offsets_m(
    gaussians,
    width: int,
    height: int,
    f_px: float,
    *,
    max_disparity: float = 0.08,
) -> Tuple[float, float]:
    """SHARP lateral offsets (X=yaw, Y=pitch) in meters."""
    from sharp.utils.camera import TrajectoryParams, compute_max_offset

    params = TrajectoryParams(type="swipe", max_disparity=max_disparity, num_steps=2)
    off = compute_max_offset(gaussians, params, (width, height), f_px)
    return float(off[0]), float(off[1])


def yaw_offset_m(
    gaussians,
    width: int,
    height: int,
    f_px: float,
    *,
    max_disparity: float = 0.08,
) -> float:
    """SHARP swipe 궤적과 동일한 최대 좌우 오프셋(m)."""
    return camera_offsets_m(
        gaussians, width, height, f_px, max_disparity=max_disparity,
    )[0]


# compute_max_offset() lateral 1× ↔ degree (SHARP swipe 단위, 상한·클램프 없음)
_OFFSET_REF_YAW_DEG = 48.0
_OFFSET_REF_PITCH_DEG = 15.0


def view_to_eye(
    yaw_deg: float,
    pitch_deg: float,
    offset_x_m: float,
    offset_y_m: float,
) -> torch.Tensor:
    """yaw/pitch° → 카메라 eye (SHARP lateral X/Y, 선형·무제한)."""
    tx = offset_x_m * float(yaw_deg) / _OFFSET_REF_YAW_DEG
    ty = offset_y_m * float(pitch_deg) / _OFFSET_REF_PITCH_DEG
    return torch.tensor([tx, ty, 0.0], dtype=torch.float32)


def yaw_to_eye(yaw_deg: float, offset_x_m: float) -> torch.Tensor:
    """yaw° → 카메라 eye position (pitch=0)."""
    return view_to_eye(yaw_deg, 0.0, offset_x_m, 0.0)


def setup_camera(gaussians, width: int, height: int, f_px: float, device: torch.device):
    """PinholeCameraModel + intrinsics."""
    from sharp.utils.camera import create_camera_model

    intrinsics = torch.tensor(
        [
            [f_px, 0, (width - 1) / 2.0, 0],
            [0, f_px, (height - 1) / 2.0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    camera_model = create_camera_model(
        gaussians, intrinsics, resolution_px=(width, height),
    )
    return camera_model


@torch.no_grad()
def render_view(
    scene: SharpScene,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    *,
    out_long: int = 1024,
    max_disparity: float = 0.08,
    offset_x_m: Optional[float] = None,
    offset_y_m: Optional[float] = None,
    gaussians=None,
    camera_model=None,
    renderer=None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """단일 yaw/pitch 시점 렌더 → (rgb uint8 H×W×3, alpha float32 H×W)."""
    device = device or require_cuda()
    width, height = output_resolution(scene, out_long)
    f_px = scene.f_px * (width / scene.width)

    if gaussians is None:
        gaussians = scene_to_gaussians(scene)
    if camera_model is None:
        camera_model = setup_camera(gaussians, width, height, f_px, device)
    if offset_x_m is None or offset_y_m is None:
        ox, oy = camera_offsets_m(
            gaussians, width, height, f_px, max_disparity=max_disparity,
        )
        if offset_x_m is None:
            offset_x_m = ox
        if offset_y_m is None:
            offset_y_m = oy

    eye = view_to_eye(yaw_deg, pitch_deg, offset_x_m, offset_y_m)
    cam = camera_model.compute(eye)

    if gsplat_cuda_ready():
        if renderer is None:
            renderer = get_renderer()
        g = gaussians.to(device)
        out = renderer(
            g,
            extrinsics=cam.extrinsics[None].to(device),
            intrinsics=cam.intrinsics[None].to(device),
            image_width=cam.width,
            image_height=cam.height,
        )
        rgb = (out.color[0].permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        alpha = out.alpha[0, 0].clamp(0, 1).cpu().numpy().astype(np.float32)
        return rgb, alpha

    import splat_torch

    g = gaussians.to(device)
    return splat_torch.render_gaussians(
        g,
        cam.extrinsics,
        cam.intrinsics,
        cam.width,
        cam.height,
        device=device,
    )


@torch.no_grad()
def render_yaw(
    scene: SharpScene,
    yaw_deg: float,
    *,
    out_long: int = 1024,
    max_disparity: float = 0.08,
    offset_x_m: Optional[float] = None,
    gaussians=None,
    camera_model=None,
    renderer=None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """yaw-only (pitch=0) 렌더."""
    return render_view(
        scene,
        yaw_deg,
        0.0,
        out_long=out_long,
        max_disparity=max_disparity,
        offset_x_m=offset_x_m,
        gaussians=gaussians,
        camera_model=camera_model,
        renderer=renderer,
        device=device,
    )
