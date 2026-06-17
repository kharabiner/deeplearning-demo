"""
reframe_iter_sharp.py — SHARP → 미세 회전 렌더 → 생성(인페인팅) → SHARP → 병합 반복

사용자 아이디어: 각 (yaw, pitch) 격자에서
  1. 현재 장면을 해당 각도로 렌더
  2. 빈 곳(~coverage)을 인페인팅으로 채움
  3. 채워진 2D 이미지를 다시 SHARP로 3D 들어올림
  4. 원본 좌표계로 변환 후 기존 Gaussian과 병합

주의: SHARP 1회 ≈ 60~90초(GPU). 32×20=640회면 수십 시간.
      --n_yaw / --n_pitch 로 격자 조절. 먼저 3×3 등으로 시험 권장.
"""

from __future__ import annotations

import argparse
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

import common
import inpaint as rinp
import splat_render
from reframe_core import CameraMove
from splat_render import _rotation
from task_nvs_sharp import SharpScene, predict


def _rotate_quats_wxyz(quats: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """쿼터니언(w,x,y,z)에 3×3 회전 R 적용."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    vx = torch.stack([x, y, z], dim=1)
    rv = (R @ vx.T).T
    return torch.stack([w, rv[:, 0], rv[:, 1], rv[:, 2]], dim=1)


def transform_scene_from_view(
    scene: SharpScene,
    move: CameraMove,
    pivot_z: float,
    ref: SharpScene,
    device: str,
) -> SharpScene:
    """회전 시점에서 SHARP한 장면 → 원본(정면) 좌표계로 변환."""
    dtype = torch.float32
    s = scene.to(device)
    pivot = torch.tensor([0.0, 0.0, pivot_z], device=device, dtype=dtype)
    R = _rotation(move.yaw_deg, move.pitch_deg, device, dtype)
    t = torch.tensor(
        [move.truck * pivot_z, move.pedestal * pivot_z, -move.dolly * pivot_z],
        device=device, dtype=dtype,
    )
    means = (s.means - pivot - t) @ R + pivot
    quats = _rotate_quats_wxyz(s.quats, R)
    return SharpScene(
        means=means, scales=s.scales, quats=quats,
        colors=s.colors, opacities=s.opacities,
        f_px=ref.f_px, width=ref.width, height=ref.height,
    )


def merge_scenes(
    base: SharpScene,
    extra: SharpScene,
    *,
    max_gaussians: int = 2_400_000,
    device: str,
) -> SharpScene:
    """두 장면을 concat. 상한 초과 시 opacity 상위만 유지."""
    means = torch.cat([base.means, extra.means], dim=0)
    scales = torch.cat([base.scales, extra.scales], dim=0)
    quats = torch.cat([base.quats, extra.quats], dim=0)
    colors = torch.cat([base.colors, extra.colors], dim=0)
    opac = torch.cat([base.opacities.reshape(-1), extra.opacities.reshape(-1)], dim=0)
    n = means.shape[0]
    if n > max_gaussians:
        idx = torch.topk(opac, max_gaussians).indices
        means, scales, quats = means[idx], scales[idx], quats[idx]
        colors, opac = colors[idx], opac[idx]
        print(f"[iter] prune {n:,} → {max_gaussians:,} gaussians")
    return SharpScene(
        means=means, scales=scales, quats=quats, colors=colors, opacities=opac,
        f_px=base.f_px, width=base.width, height=base.height,
    )


def _grid_angles(
    yaw_max: float, pitch_max: float, n_yaw: int, n_pitch: int,
) -> List[Tuple[float, float]]:
    """중앙(0,0)에서 바깥으로 퍼지는 순서."""
    yaws = np.linspace(-yaw_max, yaw_max, n_yaw)
    pitches = np.linspace(-pitch_max, pitch_max, n_pitch)
    cells = [(float(y), float(p)) for p in pitches for y in yaws]
    cells.sort(key=lambda yp: yp[0] * yp[0] + yp[1] * yp[1])
    return cells


def build_iterative_scene(
    image: Image.Image,
    *,
    yaw_max: float = 16.0,
    pitch_max: float = 10.0,
    n_yaw: int = 32,
    n_pitch: int = 20,
    backend: str = "lama",
    out_long: int = 512,
    size_boost: float = 2.2,
    skip_center: bool = True,
    max_gaussians: int = 2_400_000,
    device: Optional[str] = None,
    progress=None,
) -> SharpScene:
    device = device or common.get_device()
    img = common.resize_if_needed(image.convert("RGB"), max_size=1024)

    t0 = time.time()
    if progress:
        progress(0.02, desc="SHARP 초기 장면")
    scene = predict(img, device=device)
    ref = scene
    pivot_z = float(torch.median(scene.means[:, 2]).item())
    print(f"[iter] initial {scene.num_gaussians:,} gaussians in {time.time()-t0:.1f}s")

    if scene.width >= scene.height:
        W, H = out_long, max(1, round(out_long * scene.height / scene.width))
    else:
        H, W = out_long, max(1, round(out_long * scene.width / scene.height))

    views = _grid_angles(yaw_max, pitch_max, n_yaw, n_pitch)
    if skip_center:
        views = [(y, p) for y, p in views if abs(y) > 0.05 or abs(p) > 0.05]

    inp = rinp.get_inpainter(backend, device)
    total = len(views)
    times: List[float] = []

    for i, (yaw, pitch) in enumerate(views):
        move = CameraMove(yaw_deg=yaw, pitch_deg=pitch)
        t_step = time.time()

        rgb, cov = splat_render.render(
            scene, move, out_hw=(H, W), pivot_z=pivot_z,
            size_boost=size_boost, polish=False, device=device,
        )
        hole = ~cov
        n_hole = int(hole.sum())
        if n_hole < 50:
            print(f"[iter] {i+1}/{total} yaw={yaw:+.1f} pitch={pitch:+.1f} skip (hole {n_hole}px)")
            continue

        if progress:
            progress(0.05 + 0.9 * i / total,
                     desc=f"반복 {i+1}/{total} yaw{yaw:+.0f} pitch{pitch:+.0f} · hole {n_hole}px")

        completed = inp.inpaint(common.numpy_to_pil(rgb), hole, prompt=None)
        completed = completed.resize((W, H), Image.LANCZOS)

        t_sh = time.time()
        scene_new = predict(completed, device=device)
        dt_sh = time.time() - t_sh

        scene_new = transform_scene_from_view(scene_new, move, pivot_z, ref, device)
        scene = merge_scenes(scene, scene_new, max_gaussians=max_gaussians, device=device)

        dt = time.time() - t_step
        times.append(dt)
        avg = sum(times) / len(times)
        eta = avg * (total - i - 1)
        print(f"[iter] {i+1}/{total} yaw={yaw:+.1f} pitch={pitch:+.1f} "
              f"hole={n_hole} sharp={dt_sh:.0f}s total={dt:.0f}s "
              f"N={scene.num_gaussians:,} ETA~{eta/60:.0f}min")

    try:
        inp.unload()
    except Exception:
        pass
    common.free_memory(device)
    print(f"[iter] done {scene.num_gaussians:,} gaussians, {len(times)} steps")
    return scene


def main():
    ap = argparse.ArgumentParser(description="SHARP 반복 보완 (격자)")
    ap.add_argument("--image", default="sample2.jpg")
    ap.add_argument("--n_yaw", type=int, default=3, help="yaw 격자 수 (전체 32면 -16~16)")
    ap.add_argument("--n_pitch", type=int, default=3, help="pitch 격자 수 (전체 20면 -10~10)")
    ap.add_argument("--yaw_max", type=float, default=16.0)
    ap.add_argument("--pitch_max", type=float, default=10.0)
    ap.add_argument("--backend", default="lama", choices=["lama", "sd2", "opencv"])
    ap.add_argument("--out_long", type=int, default=512)
    ap.add_argument("--save", default="outputs/scene_iter.pt")
    ap.add_argument("--render_yaw", type=float, default=12.0)
    ap.add_argument("--render_pitch", type=float, default=8.0)
    args = ap.parse_args()

    dev = common.get_device()
    print(f"[env] {common.device_info()}")
    n_total = args.n_yaw * args.n_pitch
    print(f"[iter] grid {args.n_yaw}×{args.n_pitch} = {n_total} cells "
          f"(skip center → ~{max(0, n_total-1)} SHARP reruns)")

    img = common.load_image(args.image)
    t0 = time.time()
    scene = build_iterative_scene(
        img, n_yaw=args.n_yaw, n_pitch=args.n_pitch,
        yaw_max=args.yaw_max, pitch_max=args.pitch_max,
        backend=args.backend, out_long=args.out_long, device=dev,
    )
    elapsed = time.time() - t0
    print(f"[iter] wall {elapsed/60:.1f} min")

  # 32×20 추정
    if args.n_yaw <= 8:
        per_cell = elapsed / max(1, args.n_yaw * args.n_pitch - 1)
        est_640 = per_cell * 639 / 3600
        print(f"[iter] extrapolate 32×20 (639 steps): ~{est_640:.1f} hours "
              f"(@ {per_cell:.0f}s/step, LaMa)")

    import os
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({
        "means": scene.means.cpu(), "scales": scene.scales.cpu(),
        "quats": scene.quats.cpu(), "colors": scene.colors.cpu(),
        "opacities": scene.opacities.cpu(),
        "f_px": scene.f_px, "width": scene.width, "height": scene.height,
    }, args.save)
    print(f"[iter] saved {args.save}")

    pivot_z = float(torch.median(scene.means[:, 2]).item())
    rgb, cov = splat_render.render(
        scene, CameraMove(yaw_deg=args.render_yaw, pitch_deg=args.render_pitch),
        out_hw=(480, 640), pivot_z=pivot_z, device=dev,
    )
    common.save_result(rgb, "iter_result.png")
    print(f"[iter] render test yaw={args.render_yaw} hole% {100*(1-cov.mean()):.1f}")


if __name__ == "__main__":
    main()
