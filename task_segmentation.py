"""
task_segmentation.py — Pixel-level Segmentation
모델: SAM2 hiera base+ (facebook/sam2-hiera-base-plus)
출처: Hugging Face — transformers 라이브러리 사용 (별도 sam2 패키지 불필요)

사용법:
    # BBox 기반 (Detection 결과 연동)
    python task_segmentation.py --image sample.jpg --mode bbox

    # 이미지 중앙 포인트 클릭 예시
    python task_segmentation.py --image sample.jpg --mode point

    # 이미지 전체 자동 분할
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
from transformers import Sam2Processor, Sam2Model, pipeline

from common import (
    get_device,
    get_dtype,
    load_image,
    resize_if_needed,
    pil_to_numpy,
    numpy_to_pil,
    save_figure,
    save_result,
    free_memory,
    OUTPUTS_DIR,
)

MODEL_ID = "facebook/sam2-hiera-base-plus"


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_model(device: str):
    """SAM2 Processor + Model 로드 (HuggingFace transformers)."""
    print(f"[segmentation] Loading model: {MODEL_ID}")
    dtype = get_dtype(device)
    processor = Sam2Processor.from_pretrained(MODEL_ID)
    model = Sam2Model.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print(f"[segmentation] Model ready on {device} ({dtype})")
    return processor, model


def load_auto_pipeline(device: str):
    """자동 마스크 생성용 pipeline 로드."""
    print(f"[segmentation] Loading auto mask pipeline: {MODEL_ID}")
    pipe = pipeline(
        "mask-generation",
        model=MODEL_ID,
        device=device if device != "mps" else "cpu",  # pipeline MPS 미지원 시 CPU fallback
        torch_dtype=get_dtype(device),
    )
    print(f"[segmentation] Auto pipeline ready")
    return pipe


# ── BBox 기반 추론 ─────────────────────────────────────────────────────────────
def run_with_boxes(
    image: Image.Image,
    processor: Sam2Processor,
    model: Sam2Model,
    boxes: Union[np.ndarray, list],
    device: Optional[str] = None,
) -> list[dict]:
    """
    Bounding Box를 힌트로 SAM2 마스크 생성.

    Args:
        image: RGB PIL Image
        processor: Sam2Processor
        model: Sam2Model
        boxes: [[x1,y1,x2,y2], ...] 픽셀 절대 좌표

    Returns:
        list of {"mask": np.ndarray bool (H,W), "score": float}
    """
    if device is None:
        device = next(model.parameters()).device.type

    boxes_list = [list(map(float, b)) for b in boxes]

    results = []
    for box in boxes_list:
        # processor에 box 단위로 넘김 (배치 처리도 가능하나 메모리 안전)
        inputs = processor(
            images=image,
            input_boxes=[[box]],   # [[[x1,y1,x2,y2]]] — (batch, num_boxes, 4)
            return_tensors="pt",
        )
        # float32 강제 (MPS float64 방지)
        inputs = {
            k: v.to(device).to(torch.float32) if v.dtype == torch.float64
            else v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        masks, scores, _ = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # masks[0]: (num_masks, H, W)  scores[0]: (num_masks,)
        best_idx = scores[0].argmax().item()
        results.append({
            "mask": masks[0][best_idx].numpy().astype(bool),
            "score": float(scores[0][best_idx]),
        })

    return results


# ── 포인트 기반 추론 ───────────────────────────────────────────────────────────
def run_with_points(
    image: Image.Image,
    processor: Sam2Processor,
    model: Sam2Model,
    points: list[list[int]],
    point_labels: list[int],
    device: Optional[str] = None,
) -> list[dict]:
    """
    포인트 클릭을 힌트로 SAM2 마스크 생성.

    Args:
        points: [[x, y], ...] 픽셀 좌표
        point_labels: 1=positive(포함), 0=negative(제외)

    Returns:
        list of {"mask": np.ndarray bool (H,W), "score": float}
        (신뢰도 높은 순 정렬)
    """
    if device is None:
        device = next(model.parameters()).device.type

    inputs = processor(
        images=image,
        input_points=[points],   # (batch, num_points, 2)
        input_labels=[point_labels],
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device).to(torch.float32) if v.dtype == torch.float64
        else v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    masks, scores, _ = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    results = []
    for mask, score in zip(masks[0], scores[0]):
        results.append({
            "mask": mask.numpy().astype(bool),
            "score": float(score),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ── 자동 마스크 생성 ───────────────────────────────────────────────────────────
def run_auto(
    image: Image.Image,
    pipe,
    points_per_batch: int = 32,
) -> list[dict]:
    """
    이미지 전체 자동 세그멘테이션 (HuggingFace mask-generation pipeline).

    Returns:
        list of {"mask": np.ndarray bool (H,W), "score": float}
    """
    outputs = pipe(image, points_per_batch=points_per_batch)

    results = []
    for mask_data in outputs["masks"]:
        score = float(outputs["scores"][len(results)]) if "scores" in outputs else 1.0
        results.append({
            "mask": np.array(mask_data, dtype=bool),
            "score": score,
        })

    # 마스크 면적 큰 순 정렬 (시각화 우선순위)
    results.sort(key=lambda x: x["mask"].sum(), reverse=True)
    return results


# ── 시각화 ─────────────────────────────────────────────────────────────────────
MASK_COLORS = [
    [230, 25, 75],  [60, 180, 75],  [67, 99, 216],  [245, 130, 49],
    [145, 30, 180], [66, 212, 244], [240, 50, 230],  [191, 239, 69],
    [250, 190, 212],[70, 153, 144], [220, 190, 255], [255, 250, 200],
]


def masks_to_overlay(
    image: Image.Image,
    results: list[dict],
    alpha: float = 0.5,
) -> np.ndarray:
    """원본 이미지 위에 마스크를 반투명 색상으로 오버레이. Returns: uint8 (H,W,3)"""
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
    axes[1].set_title(f"SAM2 Segmentation — {len(results)} mask(s)", fontsize=12)

    if labels:
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
    parser = argparse.ArgumentParser(description="Pixel-level Segmentation (SAM2 via HuggingFace transformers)")
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
        pipe = load_auto_pipeline(device)
        results = run_auto(image, pipe)
        print_results(results)
        visualize(image, results, save_name=f"{stem}_seg_auto.png", show=not args.no_show)

    elif args.mode == "point":
        processor, model = load_model(device)
        center_x, center_y = W // 2, H // 2
        print(f"[segmentation] Point: ({center_x}, {center_y})")
        results = run_with_points(
            image, processor, model,
            points=[[center_x, center_y]],
            point_labels=[1],
            device=device,
        )
        print_results(results)
        visualize(image, results, save_name=f"{stem}_seg_point.png", show=not args.no_show)
