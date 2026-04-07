"""
task_detection.py — Open-Vocabulary Object Detection
모델: Grounding DINO base (IDEA-Research/grounding-dino-base)

사용법:
    python task_detection.py --image sample.jpg --prompt "person . laptop . bottle ."

동작:
    텍스트 프롬프트로 원하는 객체를 자유롭게 지정 → Bounding Box + 레이블 시각화
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from common import (
    get_device,
    get_dtype,
    load_image,
    resize_if_needed,
    pil_to_numpy,
    save_figure,
    free_memory,
    OUTPUTS_DIR,
)

MODEL_ID = "IDEA-Research/grounding-dino-base"


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_model(device: str):
    """Grounding DINO processor + model 로드."""
    print(f"[detection] Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID,
        torch_dtype=get_dtype(device),
    ).to(device)
    model.eval()
    print(f"[detection] Model ready on {device}")
    return processor, model


# ── 추론 ───────────────────────────────────────────────────────────────────────
def run(
    image: Image.Image,
    processor,
    model,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: Optional[str] = None,
) -> dict:
    """
    Grounding DINO로 객체 탐지.

    Args:
        image: RGB PIL Image
        processor: Grounding DINO processor
        model: Grounding DINO model
        text_prompt: 탐지할 객체 목록 (예: "person . laptop . bottle .")
                     — 점(.) 으로 구분, 마지막에도 점 필요
        box_threshold: 박스 신뢰도 임계값
        text_threshold: 텍스트 매칭 임계값
        device: 사용할 device (None이면 model의 device 사용)

    Returns:
        {
          "boxes":  Tensor[N, 4] (xyxy, 정규화 좌표 0~1),
          "scores": Tensor[N],
          "labels": list[str],
          "image_size": (W, H),
        }
    """
    if device is None:
        device = next(model.parameters()).device.type

    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # MPS float64 방지: post_process 내부에서 float64 텐서 생성 가능
    W, H = image.size
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(H, W)],
    )[0]

    return {
        "boxes": results["boxes"].cpu().float(),   # float32 강제 (MPS 안전)
        "scores": results["scores"].cpu().float(),
        "labels": results["labels"],
        "image_size": (W, H),
    }


# ── 시각화 ─────────────────────────────────────────────────────────────────────
COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]


def visualize(
    image: Image.Image,
    result: dict,
    save_name: str = "detection_result.png",
    show: bool = True,
) -> plt.Figure:
    """
    BBox + 레이블 + 신뢰도를 원본 이미지 위에 표시.
    """
    img_np = pil_to_numpy(image)
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img_np)
    ax.axis("off")
    ax.set_title(f"Grounding DINO — {len(boxes)} object(s) detected", fontsize=13)

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box.tolist()
        color = COLORS[i % len(COLORS)]

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(y1 - 5, 10),
            f"{label} {score:.2f}",
            color="white",
            fontsize=9,
            bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"),
        )

    plt.tight_layout()
    save_figure(fig, save_name)

    if show:
        plt.show()

    return fig


# ── 결과 요약 출력 ─────────────────────────────────────────────────────────────
def print_results(result: dict) -> None:
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]
    W, H = result["image_size"]

    print(f"\n{'─'*50}")
    print(f"Detected {len(boxes)} object(s)  [image: {W}x{H}]")
    print(f"{'─'*50}")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = [round(v, 1) for v in box.tolist()]
        print(f"  [{i+1}] {label:<20} score={score:.3f}  box=({x1},{y1})~({x2},{y2})")
    print(f"{'─'*50}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Open-Vocabulary Object Detection (Grounding DINO)")
    parser.add_argument("--image", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument(
        "--prompt", type=str,
        default="person . laptop . chair . bottle . cup . phone . bag .",
        help="탐지할 객체 (점으로 구분, 예: 'person . laptop . bottle .')",
    )
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--no-show", action="store_true", help="화면 표시 생략")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = get_device()
    print(f"[detection] Device: {device}")

    image = load_image(args.image)
    print(f"[detection] Image: {args.image}  size={image.size}")

    processor, model = load_model(device)

    result = run(
        image, processor, model,
        text_prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
    )

    print_results(result)

    stem = Path(args.image).stem
    visualize(image, result, save_name=f"{stem}_detection.png", show=not args.no_show)
