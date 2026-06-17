"""
reframe_layers.py — 사람(얼굴) 보존용 빌보드 레이어링

문제: 단일 뎁스 워핑은 얼굴 위 뎁스가 울퉁불퉁해 이목구비가 어긋나고,
      사람 실루엣 경계에서 전경이 배경으로 늘어난다(rubber-sheet).

해결(애플 Spatial Reframing 과 같은 결): 사람을 "평면 판(billboard)"으로 취급.
  1. Grounding DINO 로 'person' 검출 → SAM2 로 픽셀 마스크
  2. 사람을 지운 배경(clean plate)을 LaMa 로 1회 인페인팅 (분석 시 1회만)
  3. 드래그 시:
       - 배경: 뎁스로 워핑 (사람이 없으니 늘어남 없음)
       - 사람: 각자 median depth 의 평면으로 보고, 카메라 이동에 따라
               통째로 위치 이동 + 약간의 스케일만 적용 → 얼굴 내부 왜곡 0
  → 사람이 많은 이미지에서도 얼굴이 유지됨.

build_layers() 는 무거운 모델(검출/세그/LaMa)을 쓰므로 분석 단계에서 1회 호출.
render_layered() 는 모델 호출이 없어 실시간 드래그에 적합.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from common import get_device, free_memory, pil_to_numpy
import reframe_core as core


@dataclass
class PersonLayer:
    mask: np.ndarray            # bool (H, W)
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1
    centroid: Tuple[float, float]    # (u, v) 원본 픽셀
    median_disp: float          # [0,1], 1=가까움


@dataclass
class Layers:
    clean_rgb: np.ndarray       # uint8 (H, W, 3) — 사람 지운 배경
    clean_disp: np.ndarray      # float32 (H, W) — 사람 영역 채운 disparity
    orig_rgb: np.ndarray        # uint8 (H, W, 3) — 원본(사람 cutout 색 추출용)
    persons: List[PersonLayer] = field(default_factory=list)
    size: Tuple[int, int] = (0, 0)  # (W, H)


# ── 사람 검출 + 마스크 ──────────────────────────────────────────────────────────
SAM2_MODEL_ID = "facebook/sam2-hiera-base-plus"


def _sam2_masks(image: Image.Image, boxes: np.ndarray, device: str) -> List[np.ndarray]:
    """
    박스별 SAM2 마스크 (transformers 5.x API).
    박스마다 3개 후보 중 IoU 최고를 선택.
    """
    import torch
    from transformers import Sam2Processor, Sam2Model

    proc = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
    model = Sam2Model.from_pretrained(SAM2_MODEL_ID, torch_dtype=torch.float32).to(device).eval()

    box_list = [[float(x) for x in b] for b in boxes]
    inputs = proc(images=image, input_boxes=[box_list], return_tensors="pt")
    inputs = {
        k: (v.to(device).float() if torch.is_floating_point(v) else v.to(device))
        for k, v in inputs.items()
    }
    with torch.no_grad():
        out = model(**inputs)

    masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
    iou = out.iou_scores.cpu()[0]  # (num_boxes, num_masks)

    result = []
    for i in range(masks.shape[0]):
        best = int(iou[i].argmax())
        result.append(masks[i, best].numpy().astype(bool))

    del proc, model
    free_memory(device)
    return result


def _detect_person_masks(image: Image.Image, device: str, box_threshold: float = 0.3):
    """Grounding DINO(person) → SAM2 마스크. 무거운 모델은 쓰고 바로 해제."""
    import task_detection

    dproc, dmodel = task_detection.load_model(device)
    det = task_detection.run(
        image, dproc, dmodel,
        text_prompt="person . people . man . woman .",
        box_threshold=box_threshold, text_threshold=0.25, device=device,
    )
    del dproc, dmodel
    free_memory(device)

    boxes = det["boxes"].numpy()
    if len(boxes) == 0:
        return []

    return _sam2_masks(image, boxes, device)


def _inpaint_disparity(disp: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """사람 영역의 disparity 를 주변값으로 채워 배경이 매끄럽게 워핑되도록."""
    try:
        import cv2

        d8 = (np.clip(disp, 0, 1) * 255).astype(np.uint8)
        mask = (hole.astype(np.uint8)) * 255
        filled = cv2.inpaint(d8, mask, 5, cv2.INPAINT_NS)
        return filled.astype(np.float32) / 255.0
    except Exception:
        out = disp.copy()
        out[hole] = float(np.median(disp[~hole])) if (~hole).any() else 0.0
        return out


# ── 레이어 빌드 (분석 시 1회) ───────────────────────────────────────────────────
def build_layers(
    image: Image.Image,
    disparity: np.ndarray,
    device: Optional[str] = None,
    inpaint_backend: str = "lama",
) -> Optional[Layers]:
    """
    사람이 있으면 Layers 반환, 없으면 None(호출측은 일반 워핑으로 폴백).
    """
    device = device or get_device()
    img_np = pil_to_numpy(image)
    H, W = img_np.shape[:2]

    masks = _detect_person_masks(image, device)
    if not masks:
        print("[layers] 사람 미검출 → 단일 레이어 워핑 사용")
        return None

    # 사람 union → 살짝 팽창(실루엣 잔상 제거)
    union = np.zeros((H, W), dtype=bool)
    for m in masks:
        union |= m
    union = core.dilate_mask(union, iterations=3)

    # 배경 clean plate (사람 제거) — LaMa 1회
    import inpaint as rinp
    try:
        inp = rinp.get_inpainter(inpaint_backend, device)
        clean_pil = inp.inpaint(image, union, prompt=None)
        inp.unload()
    except Exception as e:
        print(f"[layers] clean-plate 인페인팅 실패({inpaint_backend}) → OpenCV 폴백: {e}")
        inp = rinp.get_inpainter("opencv", device)
        clean_pil = inp.inpaint(image, union, prompt=None)
    clean_rgb = pil_to_numpy(clean_pil)
    clean_disp = _inpaint_disparity(disparity, union)

    persons: List[PersonLayer] = []
    for m in masks:
        ys, xs = np.where(m)
        if len(xs) < 50:  # 너무 작은 검출 무시
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        persons.append(PersonLayer(
            mask=m,
            bbox=(x0, y0, x1, y1),
            centroid=(float(xs.mean()), float(ys.mean())),
            median_disp=float(np.median(disparity[m])),
        ))

    print(f"[layers] 사람 {len(persons)}명 빌보드 레이어 생성")
    return Layers(
        clean_rgb=clean_rgb, clean_disp=clean_disp, orig_rgb=img_np,
        persons=persons, size=(W, H),
    )


# ── 스프라이트 합성 ─────────────────────────────────────────────────────────────
def _paste_sprite(
    canvas: np.ndarray,
    src_rgb: np.ndarray,
    person: PersonLayer,
    centroid_new: Tuple[float, float],
    scale: float,
    coverage: np.ndarray,
    feather: int = 3,
) -> None:
    """사람 cutout 을 centroid_new 위치에 scale 만큼 적용해 알파 합성 (in-place)."""
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = person.bbox

    crop = src_rgb[y0:y1, x0:x1]
    alpha = person.mask[y0:y1, x0:x1].astype(np.float32)
    bh, bw = alpha.shape

    # 스케일 적용
    scale = float(np.clip(scale, 0.5, 2.0))
    nw, nh = max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))
    if (nw, nh) != (bw, bh):
        crop = np.array(Image.fromarray(crop).resize((nw, nh), Image.LANCZOS))
        alpha = np.array(
            Image.fromarray((alpha * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
        ).astype(np.float32) / 255.0

    # 경계 부드럽게 (하드 컷 방지)
    if feather > 0:
        a_img = Image.fromarray((alpha * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=feather)
        )
        alpha = np.array(a_img).astype(np.float32) / 255.0

    # centroid 가 centroid_new 에 오도록 좌상단 계산
    cu_rel = (person.centroid[0] - x0) * scale
    cv_rel = (person.centroid[1] - y0) * scale
    tlx = int(round(centroid_new[0] - cu_rel))
    tly = int(round(centroid_new[1] - cv_rel))

    # 캔버스 범위로 클립
    dx0, dy0 = max(0, tlx), max(0, tly)
    dx1, dy1 = min(W, tlx + nw), min(H, tly + nh)
    if dx0 >= dx1 or dy0 >= dy1:
        return
    sx0, sy0 = dx0 - tlx, dy0 - tly
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)

    a = alpha[sy0:sy1, sx0:sx1, None]
    c = crop[sy0:sy1, sx0:sx1].astype(np.float32)
    region = canvas[dy0:dy1, dx0:dx1].astype(np.float32)
    canvas[dy0:dy1, dx0:dx1] = (a * c + (1 - a) * region).astype(np.uint8)
    coverage[dy0:dy1, dx0:dx1] |= (alpha[sy0:sy1, sx0:sx1] > 0.3)


# ── 레이어 렌더 (실시간, 모델 없음) ─────────────────────────────────────────────
def render_layered(
    layers: Layers,
    move: core.CameraMove,
    *,
    fov_deg: float = 55.0,
    z_near: float = 1.0,
    z_far: float = 6.0,
    smooth: bool = False,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    배경 워핑 + 사람 빌보드 합성.
    smooth=True → 배경을 연결된 backward 워핑(구멍 없음)으로 → 프리뷰 우글거림 제거.
    Returns: (rendered uint8 (H,W,3), hole_mask bool (H,W))
    """
    W, H = layers.size
    pivot_z = core.median_pivot_z(layers.clean_disp, z_near, z_far)

    warped_bg, hole = core.warp_image(
        layers.clean_rgb, layers.clean_disp, move,
        fov_deg=fov_deg, z_near=z_near, z_far=z_far, pivot_z=pivot_z,
        smooth=smooth, device=device,
    )

    canvas = warped_bg.copy()
    coverage = np.zeros((H, W), dtype=bool)
    f, cx, cy = core.intrinsics(W, H, fov_deg)

    # 먼 사람부터 그려 가까운 사람이 위에 오게 (median_disp 작을수록 멈)
    for p in sorted(layers.persons, key=lambda x: x.median_disp):
        z = core.disparity_to_z_np(np.array([p.median_disp]), z_near, z_far)
        u2, v2, z2 = core.reproject_pixels_np(
            np.array([p.centroid[0]]), np.array([p.centroid[1]]), z,
            move, pivot_z, f, cx, cy,
        )
        scale = float(z[0] / (z2[0] + 1e-6))  # 가까워지면 커짐
        _paste_sprite(canvas, layers.orig_rgb, p,
                      (float(u2[0]), float(v2[0])), scale, coverage)

    # 사람으로 덮인 곳은 배경 구멍에서 제외
    hole = hole & (~coverage)
    return canvas, hole


# ── 단독 실행: 빌보드 레이어링 vs 단일 워핑 비교 ────────────────────────────────
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from common import load_image, save_result
    import task_depth

    parser = argparse.ArgumentParser(description="빌보드 레이어링 워핑 테스트")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--yaw", type=float, default=8.0)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()

    device = get_device()
    image = load_image(args.image)
    img_np = pil_to_numpy(image)

    print("[test] depth...")
    proc, model = task_depth.load_model(device)
    depth = task_depth.run(image, proc, model, device)
    del proc, model
    free_memory(device)
    disp = core.normalize_disparity(depth)

    print("[test] build layers...")
    layers = build_layers(image, disp, device)

    stem = Path(args.image).stem
    for i in range(args.frames):
        a = -args.yaw + (2 * args.yaw) * i / max(1, args.frames - 1)
        move = core.CameraMove(yaw_deg=a)
        if layers is not None:
            rendered, hole = render_layered(layers, move, device=device)
            tag = "billboard"
        else:
            rendered, hole = core.warp_image(img_np, disp, move, device=device)
            tag = "single"
        save_result(core.fill_preview(rendered, hole), f"{stem}_layer_{tag}_yaw{a:+.1f}.png")
    print("[test] 완료 → outputs/ 확인")
