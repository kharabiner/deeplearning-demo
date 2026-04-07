"""
task_segmentation.py — Pixel-level Segmentation
모델: SAM2 hiera base+ (facebook/sam2-hiera-base-plus)

사용법:
    # Detection 결과 BBox로 세그멘테이션
    python task_segmentation.py --image sample.jpg --mode bbox

    # 이미지 전체 자동 세그멘테이션
    python task_segmentation.py --image sample.jpg --mode auto

동작:
    BBox 또는 포인트 힌트 → 픽셀 단위 마스크 생성 + 시각화
"""

import argparse
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from common import (
    get_device,
    load_image,
    resize_if_needed,
    pil_to_numpy,
    numpy_to_tensor,
    save_figure,
    save_result,
    numpy_to_pil,
    free_memory,
    OUTPUTS_DIR,
)

MODEL_ID = "facebook/sam2-hiera-base-plus"


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_model(device: str) -> SAM2ImagePredictor:
    """SAM2 ImagePredictor 로드."""
    print(f"[segmentation] Loading model: {MODEL_ID}")
    sam2 = build_sam2_hf(MODEL_ID, device=device)
    predictor = SAM2ImagePredictor(sam2)
    print(f"[segmentation] Model ready on {device}")
    return predictor


def load_auto_model(device: str) -> SAM2AutomaticMaskGenerator:
    """SAM2 자동 마스크 생성기 로드."""
    print(f"[segmentation] Loading auto mask generator: {MODEL_ID}")
    sam2 = build_sam2_hf(MODEL_ID, device=device)
    generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
    )
    print(f"[segmentation] Auto mask generator ready")
    return generator


# ── BBox 기반 추론 ─────────────────────────────────────────────────────────────
def run_with_boxes(
    image: Image.Image,
    predictor: SAM2ImagePredictor,
    boxes: Union[np.ndarray, list],
) -> list[dict]:
    """
    Bounding Box를 힌트로 SAM2 마스크 생성.

    Args:
        image: RGB PIL Image
        predictor: SAM2ImagePredictor
        boxes: [[x1,y1,x2,y2], ...] 픽셀 절대 좌표

    Returns:
        list of {"mask": np.ndarray bool (H,W), "score": float}
    """
    img_np = pil_to_numpy(image)
    predictor.set_image(img_np)

    # float32 명시 (numpy 기본은 float64 → MPS 에러 방지)
    boxes_np = np.array(boxes, dtype=np.float32)

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            box=boxes_np,
            multimask_output=False,
        )

    # masks shape: (N_boxes, 1, H, W) or (1, H, W)
    if masks.ndim == 4:
        masks = masks[:, 0]  # → (N_boxes, H, W)
    elif masks.ndim == 3 and len(boxes_np) == 1:
        masks = masks[:1]    # 단일 박스

    results = []
    for i, (mask, score) in enumerate(zip(masks, scores.flatten())):
        results.append({"mask": mask.astype(bool), "score": float(score)})

    return results


# ── 포인트 기반 추론 ───────────────────────────────────────────────────────────
def run_with_points(
    image: Image.Image,
    predictor: SAM2ImagePredictor,
    points: list[list[int]],
    point_labels: list[int],
) -> list[dict]:
    """
    포인트 클릭을 힌트로 SAM2 마스크 생성.

    Args:
        points: [[x, y], ...] 픽셀 좌표
        point_labels: 1=positive(포함), 0=negative(제외)

    Returns:
        list of {"mask": np.ndarray bool (H,W), "score": float}
    """
    img_np = pil_to_numpy(image)
    predictor.set_image(img_np)

    # float32 명시 (numpy 기본은 float64 → MPS 에러 방지)
    pts_np = np.array(points, dtype=np.float32)
    lbl_np = np.array(point_labels, dtype=np.int32)

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=pts_np,
            point_labels=lbl_np,
            multimask_output=True,
        )

    results = []
    for mask, score in zip(masks, scores):
        results.append({"mask": mask.astype(bool), "score": float(score)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ── 자동 마스크 생성 ───────────────────────────────────────────────────────────
def run_auto(
    image: Image.Image,
    generator: SAM2AutomaticMaskGenerator,
) -> list[dict]:
    """
    이미지 전체 자동 세그멘테이션.

    Returns:
        list of {"mask": np.ndarray bool (H,W), "score": float, "area": int}
    """
    img_np = pil_to_numpy(image)
    raw = generator.generate(img_np)

    results = []
    for ann in sorted(raw, key=lambda x: x["area"], reverse=True):
        results.append({
            "mask": ann["segmentation"].astype(bool),
            "score": float(ann["predicted_iou"]),
            "area": int(ann["area"]),
        })

    return results


# ── 시각화 ─────────────────────────────────────────────────────────────────────
MASK_COLORS = [
    [230, 25, 75], [60, 180, 75], [67, 99, 216], [245, 130, 49],
    [145, 30, 180], [66, 212, 244], [240, 50, 230], [191, 239, 69],
    [250, 190, 212], [70, 153, 144], [220, 190, 255], [255, 250, 200],
]


def masks_to_overlay(
    image: Image.Image,
    results: list[dict],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    원본 이미지 위에 마스크를 반투명 색상으로 오버레이.
    Returns: uint8 NumPy (H, W, 3)
    """
    img_np = pil_to_numpy(image).copy()
    overlay = img_np.copy()

    for i, res in enumerate(results):
        mask = res["mask"]
        color = MASK_COLORS[i % len(MASK_COLORS)]
        overlay[mask] = color

    blended = (img_np * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return blended


def visualize(
    image: Image.Image,
    results: list[dict],
    labels: Optional[list[str]] = None,
    save_name: str = "segmentation_result.png",
    show: bool = True,
) -> plt.Figure:
    """원본 이미지 + 마스크 오버레이 나란히 표시."""
    overlay = masks_to_overlay(image, results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(pil_to_numpy(image))
    axes[0].axis("off")
    axes[0].set_title("Original", fontsize=12)

    axes[1].imshow(overlay)
    axes[1].axis("off")
    title = f"SAM2 Segmentation — {len(results)} mask(s)"
    axes[1].set_title(title, fontsize=12)

    if labels:
        img_np = pil_to_numpy(image)
        for i, (res, label) in enumerate(zip(results, labels)):
            mask = res["mask"]
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            cy, cx = int(ys.mean()), int(xs.mean())
            color = [c / 255 for c in MASK_COLORS[i % len(MASK_COLORS)]]
            axes[1].text(
                cx, cy, label,
                color="white", fontsize=9, ha="center", va="center",
                bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"),
            )

    plt.tight_layout()
    save_figure(fig, save_name)

    # 오버레이 이미지도 별도 저장
    save_result(overlay, save_name.replace(".png", "_overlay.png"))

    if show:
        plt.show()

    return fig


def print_results(results: list[dict], labels: Optional[list[str]] = None) -> None:
    print(f"\n{'─'*50}")
    print(f"Generated {len(results)} mask(s)")
    print(f"{'─'*50}")
    for i, res in enumerate(results):
        label = labels[i] if labels and i < len(labels) else f"mask_{i+1}"
        area = res["mask"].sum()
        score = res.get("score", 0.0)
        print(f"  [{i+1}] {label:<20} score={score:.3f}  area={area:,} px")
    print(f"{'─'*50}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Pixel-level Segmentation (SAM2)")
    parser.add_argument("--image", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument(
        "--mode", type=str, choices=["auto", "point"], default="auto",
        help="auto: 자동 전체 분할 / point: 이미지 중앙 포인트 예시",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = get_device()
    print(f"[segmentation] Device: {device}")

    image = load_image(args.image)
    W, H = image.size
    print(f"[segmentation] Image: {args.image}  size={image.size}")

    stem = Path(args.image).stem

    if args.mode == "auto":
        generator = load_auto_model(device)
        results = run_auto(image, generator)
        print_results(results)
        visualize(image, results, save_name=f"{stem}_seg_auto.png", show=not args.no_show)

    elif args.mode == "point":
        predictor = load_model(device)
        # 이미지 중앙을 클릭한 것으로 예시
        center_x, center_y = W // 2, H // 2
        print(f"[segmentation] Point: ({center_x}, {center_y})")
        results = run_with_points(
            image, predictor,
            points=[[center_x, center_y]],
            point_labels=[1],
        )
        print_results(results)
        visualize(image, results, save_name=f"{stem}_seg_point.png", show=not args.no_show)
