"""
reframe_ldi.py — Layered Depth Inpainting (2-레이어, iPhone Spatial Reframing 결)

문제: 단일 뎁스맵을 한 장의 연결된 면으로 워핑하면, 깊이 불연속(물체 경계)을
      가로질러 면이 늘어난다(rubber-sheet). 아이폰에는 이 늘어남이 없다.

해법(LDI): 깊이 경계 '전경 립'을 미리 지우고 그 자리에 배경을 LaMa 로 채워둔
  '배경 플레이트'를 만든다(분석 시 1회). 시점을 틀면:
    - 전경 레이어(원본+실제 뎁스)를 워핑 → 얼굴/물체가 실제로 회전
    - 전경이 가렸다 드러난 자리(디오클루전)에는 늘어난 픽셀 대신
      '미리 채워둔 배경 플레이트'를 워핑해 덮음
  → 늘어남이 사라지고, 드러난 가장자리는 "원래 거기 있던 약간 흐릿한 배경"처럼 보인다.

build_plate(): 무거운 모델(검출/세그/LaMa) 사용 → 분석 단계 1회.
render_ldi(): 모델 호출 없음(워핑 2회 + 합성) → 실시간 드래그용.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from common import get_device, pil_to_numpy
import reframe_core as core
from reframe_layers import PersonLayer, _paste_sprite


@dataclass
class Plate:
    orig_rgb: np.ndarray    # uint8 (H,W,3) — 원본(빌보드 색 추출용)
    orig_disp: np.ndarray   # float32 (H,W) — (스무딩된) 뎁스
    fg_rgb: np.ndarray      # uint8 (H,W,3) — 전경 레이어(사람 제거 → 빌보드로 대체)
    bg_rgb: np.ndarray      # uint8 (H,W,3) — 배경 플레이트(전경 립 제거+인페인팅)
    bg_disp: np.ndarray     # float32 (H,W) — 배경 뎁스(립 자리 배경값으로 채움)
    persons: List[PersonLayer] = field(default_factory=list)
    size: Tuple[int, int] = (0, 0)  # (W,H)


def _smooth_disp(disp: np.ndarray) -> np.ndarray:
    """
    면 내부의 깊이 잡음(워핑 시 '흘러내림'의 원인)을 줄이되 물체 경계는 보존.
    엣지 보존 bilateral → 가벼운 평활.
    """
    try:
        import cv2

        d = disp.astype(np.float32)
        s = cv2.bilateralFilter(d, d=7, sigmaColor=0.08, sigmaSpace=7)
        return s
    except Exception:
        return disp


# ── 깊이 경계 → 전경 립 마스크 ──────────────────────────────────────────────────
def _edge_mask(disp: np.ndarray, q: float = 0.93) -> np.ndarray:
    """disparity 의 그래디언트 크기 상위 분위 = 깊이 불연속(물체 경계)."""
    gy, gx = np.gradient(disp.astype(np.float32))
    g = np.sqrt(gx * gx + gy * gy)
    thr = float(np.quantile(g, q))
    return g > max(thr, 1e-4)


def _foreground_lip(disp: np.ndarray, persons_union: Optional[np.ndarray],
                    band: int = 6, near_pct: float = 55.0) -> np.ndarray:
    """
    워핑 시 늘어남을 만드는 '전경 쪽' 경계 밴드.
    = (깊이 경계 팽창) ∩ (가까운 픽셀)  ∪  사람 영역.
    이 영역을 배경으로 인페인팅하면, 시점 이동 시 그 뒤의 배경이 드러난다.
    """
    edges = _edge_mask(disp)
    band_mask = core.dilate_mask(edges, iterations=band)
    near = disp >= np.percentile(disp, near_pct)
    lip = band_mask & near
    if persons_union is not None:
        lip = lip | persons_union
    return core.dilate_mask(lip, iterations=2)


def _inpaint_disp_far(disp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    립 자리의 disparity 를 '배경(먼 쪽)' 값으로 채운다.
    Telea 로 주변값 전파 후, 마스크 내부는 원본보다 멀어지도록 살짝 낮춰
    배경 플레이트가 전경보다 뒤에 위치하게 한다.
    """
    try:
        import cv2

        d8 = (np.clip(disp, 0, 1) * 255).astype(np.uint8)
        m = (mask.astype(np.uint8)) * 255
        filled = cv2.inpaint(d8, m, 5, cv2.INPAINT_NS).astype(np.float32) / 255.0
    except Exception:
        filled = disp.copy()
        bg_val = float(np.percentile(disp[~mask], 35)) if (~mask).any() else 0.0
        filled[mask] = bg_val
    out = disp.copy()
    out[mask] = filled[mask]
    return out


# ── 플레이트 빌드 (분석 시 1회) ─────────────────────────────────────────────────
def build_plate(
    image: Image.Image,
    disparity: np.ndarray,
    device: Optional[str] = None,
    inpaint_backend: str = "lama",
    detect_persons: bool = True,
) -> Plate:
    """전경 립을 지운 배경 플레이트를 인페인팅으로 미리 생성."""
    device = device or get_device()
    img_np = pil_to_numpy(image)
    H, W = img_np.shape[:2]
    disp_s = _smooth_disp(disparity)

    persons_union = None
    persons: List[PersonLayer] = []
    if detect_persons:
        try:
            import reframe_layers as rlayers

            masks = rlayers._detect_person_masks(image, device)
            if masks:
                u = np.zeros((H, W), dtype=bool)
                for m in masks:
                    u |= m
                persons_union = core.dilate_mask(u, iterations=3)
                for m in masks:
                    ys, xs = np.where(m)
                    if len(xs) < 50:
                        continue
                    persons.append(PersonLayer(
                        mask=m,
                        bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                        centroid=(float(xs.mean()), float(ys.mean())),
                        median_disp=float(np.median(disp_s[m])),
                    ))
                print(f"[ldi] 사람 {len(persons)}명 → 빌보드(머리 보존)")
        except Exception as e:
            print(f"[ldi] 사람 검출 생략: {e}")

    lip = _foreground_lip(disp_s, persons_union)

    import inpaint as rinp
    try:
        inp = rinp.get_inpainter(inpaint_backend, device)
        bg_pil = inp.inpaint(image, lip, prompt=None)
        inp.unload()
    except Exception as e:
        print(f"[ldi] 배경 인페인팅 실패({inpaint_backend}) → OpenCV 폴백: {e}")
        inp = rinp.get_inpainter("opencv", device)
        bg_pil = inp.inpaint(image, lip, prompt=None)

    bg_rgb = pil_to_numpy(bg_pil)
    bg_disp = _inpaint_disp_far(disp_s, lip)

    # 전경 레이어에서 사람 제거(빌보드로 대체) → 워핑된 머리 speckle 방지
    fg_rgb = img_np.copy()
    if persons_union is not None:
        fg_rgb[persons_union] = bg_rgb[persons_union]

    print(f"[ldi] 배경 플레이트 생성 (립 픽셀 {int(lip.sum())})")
    return Plate(
        orig_rgb=img_np, orig_disp=disp_s,
        fg_rgb=fg_rgb, bg_rgb=bg_rgb, bg_disp=bg_disp,
        persons=persons, size=(W, H),
    )


# ── 레이어 렌더 (실시간, 모델 없음) ─────────────────────────────────────────────
def _feather(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """bool 마스크 → [0,1] 부드러운 알파 (합성 경계 하드컷 방지)."""
    a = Image.fromarray((mask.astype(np.uint8) * 255)).filter(
        ImageFilter.GaussianBlur(radius=radius)
    )
    return np.asarray(a, dtype=np.float32) / 255.0


def render_ldi(
    plate: Plate,
    move: core.CameraMove,
    *,
    fov_deg: float = 55.0,
    z_near: float = 1.0,
    z_far: float = 6.0,
    frame_zoom: float = 1.0,
    smooth: bool = True,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    전경 워핑 + (안쪽 디오클루전 자리에) 배경 플레이트 합성 + 사람 빌보드.
    frame_zoom>1 = 확장(Extend): 화면 축소 → 프레임 밖 노출.

    Returns:
        rendered: uint8 (H,W,3)
        outer:    bool (H,W) — 원본 프레임 '밖'(드래그 중 블러 / 완료 시 아웃페인팅)
        residual: bool (H,W) — 안쪽 잔여 구멍(확정 인페인팅 보강용)
    """
    W, H = plate.size
    # 두 레이어가 같은 카메라/피벗을 쓰도록 pivot 고정
    pivot_z = core.median_pivot_z(plate.orig_disp, z_near, z_far)
    kw = dict(fov_deg=fov_deg, z_near=z_near, z_far=z_far, pivot_z=pivot_z,
              smooth=smooth, frame_zoom=frame_zoom, return_outer=True, device=device)

    fg, fg_hole, outer = core.warp_image(plate.fg_rgb, plate.orig_disp, move, **kw)
    bg, bg_hole, _ = core.warp_image(plate.bg_rgb, plate.bg_disp, move, **kw)

    # 안쪽 늘어난(디오클루전) 자리에만 배경 플레이트를 덮음 (바깥 프레임은 제외)
    reveal = core.dilate_mask(fg_hole, iterations=2) & (~outer)
    alpha = _feather(reveal, radius=3)[..., None]
    out = (fg.astype(np.float32) * (1 - alpha) + bg.astype(np.float32) * alpha)
    out = out.clip(0, 255).astype(np.uint8)

    # 사람: 빌보드로 재투영해 위에 합성 (머리 내부 왜곡 0)
    coverage = np.zeros((H, W), dtype=bool)
    f, cx, cy = core.intrinsics(W, H, fov_deg)
    # 확장 시 사람도 같은 비율로 축소되도록 중심 기준 스케일 보정
    inv_zoom = 1.0 / frame_zoom
    for p in sorted(plate.persons, key=lambda x: x.median_disp):
        z = core.disparity_to_z_np(np.array([p.median_disp]), z_near, z_far)
        u2, v2, z2 = core.reproject_pixels_np(
            np.array([p.centroid[0]]), np.array([p.centroid[1]]), z,
            move, pivot_z, f, cx, cy,
        )
        scale = float(z[0] / (z2[0] + 1e-6)) * inv_zoom
        # frame_zoom 이 적용된 화면 좌표로 변환 (중심 기준 축소)
        uu2 = (W - 1) * 0.5 + (float(u2[0]) - (W - 1) * 0.5) * inv_zoom
        vv2 = (H - 1) * 0.5 + (float(v2[0]) - (H - 1) * 0.5) * inv_zoom
        _paste_sprite(out, plate.orig_rgb, p, (uu2, vv2), scale, coverage)

    residual = (fg_hole & bg_hole) & (~coverage) & (~outer)
    return out, outer, residual


# ── 단독 실행: LDI vs 단일 워핑 비교 ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from common import load_image, save_result, free_memory
    import task_depth_depthanythingv2 as task_depth

    parser = argparse.ArgumentParser(description="LDI 워핑 테스트")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--yaw", type=float, default=8.0)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--zoom", type=float, default=1.0, help="Extend(확장) 배율 >1")
    parser.add_argument("--no-persons", action="store_true")
    args = parser.parse_args()

    device = get_device()
    image = load_image(args.image)

    print("[test] depth...")
    proc, model = task_depth.load_model(device)
    depth = task_depth.run(image, proc, model, device)
    del proc, model
    free_memory(device)
    disp = core.normalize_disparity(depth)

    print("[test] build plate...")
    plate = build_plate(image, disp, device, detect_persons=not args.no_persons)

    stem = Path(args.image).stem
    for i in range(args.frames):
        a = -args.yaw + (2 * args.yaw) * i / max(1, args.frames - 1)
        out, outer, _ = render_ldi(
            plate, core.CameraMove(yaw_deg=a), frame_zoom=args.zoom, device=device,
        )
        # 드래그 프리뷰처럼 바깥 프레임은 블러로 표시
        out = core.fill_preview(out, outer)
        save_result(out, f"{stem}_ldi_yaw{a:+.1f}_z{args.zoom:.2f}.png")
    print("[test] 완료 → outputs/ 확인")
