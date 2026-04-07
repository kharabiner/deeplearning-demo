"""
task_depth.py — Monocular Depth Estimation
모델: Depth Anything V2 Small (depth-anything/Depth-Anything-V2-Small-hf)

사용법:
    python task_depth.py --image sample.jpg

동작:
    이미지 1장 → 각 픽셀의 상대적 깊이(거리) 추정 → 컬러 깊이 맵 시각화
    가까울수록 따뜻한 색(빨강), 멀수록 차가운 색(파랑)
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

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

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_model(device: str):
    """Depth Anything V2 processor + model 로드."""
    print(f"[depth] Loading model: {MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_ID,
        torch_dtype=get_dtype(device),
    ).to(device)
    model.eval()
    print(f"[depth] Model ready on {device}")
    return processor, model


# ── 추론 ───────────────────────────────────────────────────────────────────────
def run(
    image: Image.Image,
    processor,
    model,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    깊이 추정 실행.

    Args:
        image: RGB PIL Image
        processor: Depth Anything processor
        model: Depth Anything model
        device: 사용 device

    Returns:
        depth_map: float32 NumPy (H, W), 값이 클수록 먼 거리 (상대적 깊이)
    """
    if device is None:
        device = next(model.parameters()).device.type

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth  # (1, H', W')

    # 원본 이미지 크기로 업샘플
    W, H = image.size
    depth_resized = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze()  # → (H, W)

    # .float() → float32 강제 (MPS에서 float64 업캐스팅 방지)
    depth_np = depth_resized.cpu().float().numpy().astype(np.float32)
    return depth_np


# ── 시각화 ─────────────────────────────────────────────────────────────────────
def depth_to_colormap(
    depth: np.ndarray,
    colormap: str = "plasma",
    invert: bool = True,
) -> np.ndarray:
    """
    깊이 맵을 컬러맵 이미지로 변환.

    Depth Anything V2: 값이 클수록 가까운 거리 (inverse depth)
    plasma colormap: 값이 클수록 warm(노랑/밝음)
    → invert=True(기본): 가까운(큰값)을 뒤집어 cool로, 먼 거리가 warm → warm=far ✓
    → invert=False: warm=near (직관과 반대)

    Returns:
        uint8 NumPy (H, W, 3)
    """
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        normalized = np.zeros_like(depth)
    else:
        normalized = (depth - d_min) / (d_max - d_min)

    if invert:
        normalized = 1.0 - normalized

    cmap = plt.get_cmap(colormap)
    colored = cmap(normalized)[:, :, :3]  # (H, W, 3) float [0,1]
    return (colored * 255).astype(np.uint8)


def visualize(
    image: Image.Image,
    depth: np.ndarray,
    colormap: str = "plasma",
    save_name: str = "depth_result.png",
    show: bool = True,
) -> plt.Figure:
    """
    원본과 depth map을 동일한 크기로 나란히 표시.
    colorbar는 depth map 오른쪽에 별도로 붙임 (이미지 크기에 영향 없음).
    """
    img_np = pil_to_numpy(image)
    H, W = img_np.shape[:2]

    # 이미지 비율에 맞춰 figure 크기 계산
    # 원본 + depth 두 패널 → 가로는 2배, 세로는 이미지 비율 유지
    dpi = 100
    fig_w = (W * 2) / dpi + 1.5   # +1.5 는 colorbar 여유
    fig_h = H / dpi

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.05})

    # 원본
    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title("Original", fontsize=11)

    # raw depth — 이미지와 동일 크기로 표시
    im = axes[1].imshow(depth, cmap=colormap,
                        extent=[0, W, H, 0],   # 픽셀 좌표로 고정
                        aspect="auto")
    axes[1].axis("off")
    axes[1].set_title("Depth Map (raw, larger = nearer)", fontsize=11)

    # colorbar를 depth 축 오른쪽에만 붙임 (원본 축 크기 변화 없음)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="4%", pad=0.05)
    plt.colorbar(im, cax=cax, label="Relative Depth")

    fig.suptitle("Depth Anything V2 — Monocular Depth Estimation", fontsize=12)
    plt.tight_layout()

    save_figure(fig, save_name)

    if show:
        plt.show()

    return fig


def print_results(depth: np.ndarray, image_size: tuple) -> None:
    W, H = image_size
    d_min = depth.min()
    d_max = depth.max()
    d_mean = depth.mean()

    print(f"\n{'─'*50}")
    print(f"Depth Estimation Result  [image: {W}x{H}]")
    print(f"{'─'*50}")
    print(f"  Min depth : {d_min:.4f}")
    print(f"  Max depth : {d_max:.4f}")
    print(f"  Mean depth: {d_mean:.4f}")
    print(f"  (Values are relative — larger = nearer)")
    print(f"{'─'*50}\n")


# ── 객체별 평균 깊이 계산 (파이프라인 연동용) ──────────────────────────────────
def get_depth_for_masks(
    depth: np.ndarray,
    masks: list[np.ndarray],
    labels: Optional[list[str]] = None,
) -> list[dict]:
    """
    세그멘테이션 마스크별 평균/중앙값 깊이 계산.
    pipeline_final.py에서 Detection → Segmentation → Depth 연동 시 사용.

    Args:
        depth: float32 NumPy (H, W)
        masks: bool NumPy (H, W) 마스크 리스트
        labels: 마스크별 레이블 이름

    Returns:
        [{"label": str, "mean_depth": float, "median_depth": float, "rank": int}, ...]
        rank=1이 가장 가까운 객체
    """
    results = []
    for i, mask in enumerate(masks):
        label = labels[i] if labels and i < len(labels) else f"object_{i+1}"
        pixel_depths = depth[mask]
        if len(pixel_depths) == 0:
            continue
        results.append({
            "label": label,
            "mean_depth": float(pixel_depths.mean()),
            "median_depth": float(np.median(pixel_depths)),
        })

    # 가장 가까운 객체 = depth 값이 작은 것
    results.sort(key=lambda x: x["mean_depth"])
    for rank, r in enumerate(results, 1):
        r["rank"] = rank  # 1=가장 가까움

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Monocular Depth Estimation (Depth Anything V2)")
    parser.add_argument("--image", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument("--colormap", type=str, default="plasma",
                        choices=["plasma", "inferno", "magma", "viridis"],
                        help="raw depth map 컬러맵 (default: plasma)")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = get_device()
    print(f"[depth] Device: {device}")

    image = load_image(args.image)
    print(f"[depth] Image: {args.image}  size={image.size}")

    processor, model = load_model(device)
    depth = run(image, processor, model, device)

    print_results(depth, image.size)

    stem = Path(args.image).stem
    visualize(image, depth, colormap=args.colormap,
              save_name=f"{stem}_depth.png", show=not args.no_show)
