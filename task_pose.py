"""
task_pose.py — Human Pose Estimation
모델: ViTPose base (usyd-community/vitpose-base-simple)

사용법:
    python task_pose.py --image sample.jpg

동작:
    사람이 포함된 이미지 → 관절 17개 키포인트 추정 → 스켈레톤 시각화

COCO 17 keypoints 순서:
    0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
    5:left_shoulder  6:right_shoulder  7:left_elbow  8:right_elbow
    9:left_wrist  10:right_wrist  11:left_hip  12:right_hip
    13:left_knee  14:right_knee  15:left_ankle  16:right_ankle
"""

import argparse
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

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

MODEL_ID = "usyd-community/vitpose-base-simple"

# COCO 17 keypoint 이름
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 스켈레톤 연결선 (키포인트 인덱스 쌍)
SKELETON_EDGES = [
    (0, 1), (0, 2),          # 코 → 눈
    (1, 3), (2, 4),          # 눈 → 귀
    (0, 5), (0, 6),          # 코 → 어깨 (근사)
    (5, 6),                  # 좌어깨 ↔ 우어깨
    (5, 7), (7, 9),          # 왼팔
    (6, 8), (8, 10),         # 오른팔
    (5, 11), (6, 12),        # 어깨 → 엉덩이
    (11, 12),                # 엉덩이
    (11, 13), (13, 15),      # 왼다리
    (12, 14), (14, 16),      # 오른다리
]

# 부위별 색상
LEFT_COLOR  = "#4363d8"   # 파랑 (왼쪽)
RIGHT_COLOR = "#e6194b"   # 빨강 (오른쪽)
CENTER_COLOR = "#3cb44b"  # 초록 (중앙)

EDGE_COLORS = {
    (0, 1): CENTER_COLOR, (0, 2): CENTER_COLOR,
    (1, 3): LEFT_COLOR,   (2, 4): RIGHT_COLOR,
    (0, 5): LEFT_COLOR,   (0, 6): RIGHT_COLOR,
    (5, 6): CENTER_COLOR,
    (5, 7): LEFT_COLOR,   (7, 9): LEFT_COLOR,
    (6, 8): RIGHT_COLOR,  (8, 10): RIGHT_COLOR,
    (5, 11): LEFT_COLOR,  (6, 12): RIGHT_COLOR,
    (11, 12): CENTER_COLOR,
    (11, 13): LEFT_COLOR, (13, 15): LEFT_COLOR,
    (12, 14): RIGHT_COLOR,(14, 16): RIGHT_COLOR,
}


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_model(device: str):
    """ViTPose processor + model 로드."""
    print(f"[pose] Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=get_dtype(device),
    ).to(device)
    model.eval()
    print(f"[pose] Model ready on {device}")
    return processor, model


# ── 추론 ───────────────────────────────────────────────────────────────────────
def run(
    image: Image.Image,
    processor,
    model,
    person_boxes: Optional[list] = None,
    score_threshold: float = 0.3,
    device: Optional[str] = None,
) -> list[dict]:
    """
    사람 키포인트 추정.

    Args:
        image: RGB PIL Image
        processor: ViTPose processor
        model: ViTPose model
        person_boxes: [[x1,y1,x2,y2], ...] 사람 BBox 목록
                      None이면 이미지 전체를 단일 사람으로 처리
        score_threshold: 키포인트 신뢰도 임계값
        device: 사용 device

    Returns:
        [
          {
            "keypoints": np.ndarray (17, 2) — (x, y) 픽셀 좌표,
            "scores":    np.ndarray (17,)   — 키포인트별 신뢰도,
            "box":       [x1, y1, x2, y2]  — 사람 BBox,
          },
          ...
        ]
    """
    if device is None:
        device = next(model.parameters()).device.type

    W, H = image.size

    # BBox가 없으면 이미지 전체를 사람 영역으로 사용
    if person_boxes is None or len(person_boxes) == 0:
        person_boxes = [[0, 0, W, H]]

    results = []
    for box in person_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        # 사람 영역 크롭
        crop = image.crop((x1, y1, x2, y2))

        inputs = processor(images=crop, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # heatmap → keypoints
        heatmaps = outputs.heatmaps  # (1, 17, H', W')
        heatmap_np = heatmaps[0].cpu().float().numpy()  # (17, H', W')

        keypoints = []
        scores = []
        crop_w, crop_h = crop.size
        _, hm_h, hm_w = heatmap_np.shape

        for kp_heatmap in heatmap_np:
            # heatmap에서 argmax로 위치 추출
            score = float(kp_heatmap.max())
            flat_idx = int(kp_heatmap.argmax())
            kp_y = flat_idx // hm_w
            kp_x = flat_idx % hm_w

            # 크롭 좌표 → 원본 이미지 좌표로 변환
            orig_x = x1 + kp_x * crop_w / hm_w
            orig_y = y1 + kp_y * crop_h / hm_h

                keypoints.append([float(orig_x), float(orig_y)])  # float32 명시
            scores.append(float(score))

        results.append({
            "keypoints": np.array(keypoints, dtype=np.float32),  # float32 명시
            "scores": np.array(scores, dtype=np.float32),         # float32 명시
            "box": [x1, y1, x2, y2],
        })

    return results


# ── 시각화 ─────────────────────────────────────────────────────────────────────
def visualize(
    image: Image.Image,
    results: list[dict],
    score_threshold: float = 0.3,
    save_name: str = "pose_result.png",
    show: bool = True,
) -> plt.Figure:
    """스켈레톤 + 키포인트를 원본 이미지 위에 시각화."""
    img_np = pil_to_numpy(image)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title("Original", fontsize=12)

    axes[1].imshow(img_np)
    axes[1].axis("off")
    n_people = len(results)
    axes[1].set_title(f"ViTPose — {n_people} person(s)", fontsize=12)

    for res in results:
        kps = res["keypoints"]    # (17, 2)
        scs = res["scores"]       # (17,)
        box = res["box"]

        # BBox
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor="yellow", facecolor="none",
        )
        axes[1].add_patch(rect)

        # 스켈레톤 연결선
        for (i, j) in SKELETON_EDGES:
            if scs[i] < score_threshold or scs[j] < score_threshold:
                continue
            x_vals = [kps[i, 0], kps[j, 0]]
            y_vals = [kps[i, 1], kps[j, 1]]
            color = EDGE_COLORS.get((i, j), CENTER_COLOR)
            axes[1].plot(x_vals, y_vals, "-", color=color, linewidth=2, alpha=0.8)

        # 키포인트 점
        for idx, (x, y) in enumerate(kps):
            if scs[idx] < score_threshold:
                continue
            axes[1].plot(x, y, "o", color="white", markersize=5, zorder=5)
            axes[1].plot(x, y, "o", color=CENTER_COLOR, markersize=3, zorder=6)

    plt.tight_layout()
    save_figure(fig, save_name)

    if show:
        plt.show()

    return fig


def print_results(results: list[dict], score_threshold: float = 0.3) -> None:
    print(f"\n{'─'*50}")
    print(f"Pose Estimation — {len(results)} person(s) detected")
    print(f"{'─'*50}")
    for p_idx, res in enumerate(results):
        kps = res["keypoints"]
        scs = res["scores"]
        visible = int((scs >= score_threshold).sum())
        print(f"  [Person {p_idx+1}]  visible keypoints: {visible}/17")
        for k_idx, name in enumerate(KEYPOINT_NAMES):
            if scs[k_idx] >= score_threshold:
                x, y = kps[k_idx]
                print(f"    {name:<18} ({x:6.1f}, {y:6.1f})  score={scs[k_idx]:.2f}")
    print(f"{'─'*50}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Human Pose Estimation (ViTPose)")
    parser.add_argument("--image", type=str, required=True, help="입력 이미지 경로 (사람이 포함된 이미지)")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                        help="키포인트 신뢰도 임계값")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = get_device()
    print(f"[pose] Device: {device}")

    image = load_image(args.image)
    print(f"[pose] Image: {args.image}  size={image.size}")

    processor, model = load_model(device)

    # 단독 실행 시 이미지 전체를 사람 영역으로 가정
    results = run(image, processor, model, device=device,
                  score_threshold=args.score_threshold)

    print_results(results, score_threshold=args.score_threshold)

    stem = Path(args.image).stem
    visualize(image, results, score_threshold=args.score_threshold,
              save_name=f"{stem}_pose.png", show=not args.no_show)
