"""
pipeline_final.py — Final Project Pipeline (기말 프로젝트용)
"AI 현장 감식관" — 이미지 한 장에서 단계적 분석

사용법:
    python pipeline_final.py --image sample.jpg

파이프라인:
    1. Detection  (Grounding DINO) → 객체 BBox 목록
    2. Segmentation (SAM2)         → 픽셀 마스크
    3. Depth (Depth Anything V2)   → 객체별 거리 정보
    4. Pose (ViTPose)              → 사람 자세 분석 (사람이 있을 경우)
    5. VLM  (Qwen2-VL-2B)         → 분석 결과 종합 보고서

각 단계의 출력이 다음 단계의 입력으로 전달됩니다.
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from common import (
    get_device, load_image, pil_to_numpy,
    save_figure, save_result, free_memory,
    OUTPUTS_DIR,
)

# 각 태스크 모듈 import
import task_detection as det
import task_segmentation as seg
import task_depth as dep
import task_pose as pose
import task_vlm as vlm


# ── 기본 탐지 프롬프트 ─────────────────────────────────────────────────────────
DEFAULT_PROMPT = (
    "person . laptop . chair . table . bottle . cup . phone . "
    "bag . book . monitor . keyboard . mouse . plant . window ."
)


# ── 파이프라인 실행 ────────────────────────────────────────────────────────────
def run_pipeline(
    image: Image.Image,
    detection_prompt: str = DEFAULT_PROMPT,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    score_threshold: float = 0.3,
    max_vlm_tokens: int = 512,
    device: Optional[str] = None,
) -> dict:
    """
    전체 파이프라인 실행.

    Returns:
        {
          "detection":   {"boxes", "scores", "labels", "image_size"},
          "segmentation": [{"mask", "score"}, ...],
          "depth":        np.ndarray (H, W),
          "depth_per_obj": [{"label", "mean_depth", "rank"}, ...],
          "pose":         [{"keypoints", "scores", "box"}, ...],
          "vlm_report":   str,
        }
    """
    if device is None:
        device = get_device()

    W, H = image.size
    print(f"\n{'═'*60}")
    print(f"AI Scene Analyst Pipeline")
    print(f"Image: {W}x{H}  |  Device: {device}")
    print(f"{'═'*60}")

    results = {}

    # ── Step 1: Detection ──────────────────────────────────────────────────────
    print("\n[Step 1/5] Open-Vocabulary Detection (Grounding DINO)")
    det_processor, det_model = det.load_model(device)
    det_result = det.run(
        image, det_processor, det_model,
        text_prompt=detection_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
    results["detection"] = det_result
    det.print_results(det_result)

    # 메모리 해제
    del det_model
    free_memory(device)

    boxes = det_result["boxes"].numpy().tolist()   # [[x1,y1,x2,y2], ...]
    labels = det_result["labels"]

    if len(boxes) == 0:
        print("[warning] No objects detected. Adjusting thresholds or changing prompt recommended.")
        boxes = [[0, 0, W, H]]
        labels = ["unknown"]

    # ── Step 2: Segmentation ───────────────────────────────────────────────────
    print("\n[Step 2/5] Pixel-level Segmentation (SAM2)")
    seg_predictor = seg.load_model(device)
    seg_results = seg.run_with_boxes(image, seg_predictor, boxes)
    results["segmentation"] = seg_results
    seg.print_results(seg_results, labels=labels)

    del seg_predictor
    free_memory(device)

    # ── Step 3: Depth Estimation ───────────────────────────────────────────────
    print("\n[Step 3/5] Monocular Depth Estimation (Depth Anything V2)")
    dep_processor, dep_model = dep.load_model(device)
    depth_map = dep.run(image, dep_processor, dep_model, device)
    results["depth"] = depth_map
    dep.print_results(depth_map, image.size)

    # 각 객체의 마스크별 평균 깊이 계산
    masks_np = [r["mask"] for r in seg_results]
    depth_per_obj = dep.get_depth_for_masks(depth_map, masks_np, labels=labels)
    results["depth_per_obj"] = depth_per_obj

    print("  Object depth ranking (1=nearest):")
    for obj in depth_per_obj:
        print(f"    [{obj['rank']}] {obj['label']:<20} mean_depth={obj['mean_depth']:.4f}")

    del dep_model
    free_memory(device)

    # ── Step 4: Pose Estimation (사람이 있을 때만) ─────────────────────────────
    person_boxes = [
        box for box, label in zip(boxes, labels) if "person" in label.lower()
    ]

    if person_boxes:
        print(f"\n[Step 4/5] Pose Estimation (ViTPose) — {len(person_boxes)} person(s)")
        pose_processor, pose_model = pose.load_model(device)
        pose_results = pose.run(
            image, pose_processor, pose_model,
            person_boxes=person_boxes,
            score_threshold=score_threshold,
            device=device,
        )
        results["pose"] = pose_results
        pose.print_results(pose_results, score_threshold=score_threshold)

        del pose_model
        free_memory(device)
    else:
        print("\n[Step 4/5] Pose Estimation — skipped (no person detected)")
        results["pose"] = []

    # ── Step 5: VLM 종합 보고서 ───────────────────────────────────────────────
    print("\n[Step 5/5] Scene Report (Qwen2-VL-2B)")
    vlm_processor, vlm_model = vlm.load_model(device)

    context = _build_context(det_result, depth_per_obj, results["pose"], score_threshold)
    question = (
        f"Here is the analysis result of this image:\n{context}\n\n"
        "Based on this analysis and what you see in the image, please provide:\n"
        "1. A comprehensive description of the scene\n"
        "2. The spatial arrangement of objects (what is closer/farther)\n"
        "3. Any notable activities or interactions between objects/people\n"
        "4. Any unusual or interesting observations"
    )

    report = vlm.run(image, vlm_processor, vlm_model, question, max_vlm_tokens, device)
    results["vlm_report"] = report

    del vlm_model
    free_memory(device)

    print(f"\n{'─'*50}")
    print("VLM Scene Report:")
    print(f"{'─'*50}")
    print(report)
    print(f"{'─'*50}\n")

    return results


def _build_context(det_result, depth_per_obj, pose_results, score_threshold) -> str:
    """각 단계 결과를 VLM 프롬프트용 텍스트로 변환."""
    lines = []

    # 탐지된 객체
    labels = det_result["labels"]
    scores = det_result["scores"].tolist()
    lines.append(f"Detected objects ({len(labels)}):")
    for label, score in zip(labels, scores):
        lines.append(f"  - {label} (confidence: {score:.2f})")

    # 거리 정보
    if depth_per_obj:
        lines.append("\nObject distance ranking (1=nearest to camera):")
        for obj in depth_per_obj:
            lines.append(f"  {obj['rank']}. {obj['label']}")

    # 자세 정보
    if pose_results:
        lines.append(f"\nPose analysis — {len(pose_results)} person(s) detected:")
        for i, res in enumerate(pose_results):
            visible = int((res["scores"] >= score_threshold).sum())
            lines.append(f"  Person {i+1}: {visible}/17 keypoints visible")

    return "\n".join(lines)



# ── 통합 시각화 ────────────────────────────────────────────────────────────────
def visualize_all(
    image: Image.Image,
    results: dict,
    save_name: str = "pipeline_result.png",
    show: bool = True,
) -> plt.Figure:
    """
    5개 결과를 하나의 Figure에 레이아웃.
    """
    det_result = results["detection"]
    seg_results = results["segmentation"]
    depth_map   = results["depth"]
    pose_results = results["pose"]
    labels = det_result["labels"]

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)

    # ① 원본
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(pil_to_numpy(image))
    ax_orig.axis("off")
    ax_orig.set_title("① Original", fontsize=12, fontweight="bold")

    # ② Detection
    ax_det = fig.add_subplot(gs[0, 1])
    ax_det.imshow(pil_to_numpy(image))
    ax_det.axis("off")
    ax_det.set_title(f"② Detection ({len(det_result['boxes'])} objects)", fontsize=12, fontweight="bold")
    import matplotlib.patches as mpatches
    for i, (box, score, label) in enumerate(zip(
        det_result["boxes"], det_result["scores"], det_result["labels"]
    )):
        x1, y1, x2, y2 = box.tolist()
        color = det.COLORS[i % len(det.COLORS)]
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor=color, facecolor="none")
        ax_det.add_patch(rect)
        ax_det.text(x1, max(y1-5, 10), f"{label} {score:.2f}",
                    color="white", fontsize=7,
                    bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none"))

    # ③ Segmentation
    ax_seg = fig.add_subplot(gs[0, 2])
    overlay = seg.masks_to_overlay(image, seg_results)
    ax_seg.imshow(overlay)
    ax_seg.axis("off")
    ax_seg.set_title(f"③ Segmentation ({len(seg_results)} masks)", fontsize=12, fontweight="bold")

    # ④ Depth
    ax_dep = fig.add_subplot(gs[1, 0])
    depth_colored = dep.depth_to_colormap(depth_map, colormap="plasma")
    ax_dep.imshow(depth_colored)
    ax_dep.axis("off")
    ax_dep.set_title("④ Depth Map (warm=near)", fontsize=12, fontweight="bold")

    # ⑤ Pose
    ax_pose = fig.add_subplot(gs[1, 1])
    ax_pose.imshow(pil_to_numpy(image))
    ax_pose.axis("off")
    n_p = len(pose_results)
    ax_pose.set_title(f"⑤ Pose ({n_p} person)", fontsize=12, fontweight="bold")
    for res in pose_results:
        kps = res["keypoints"]
        scs = res["scores"]
        for (i, j) in pose.SKELETON_EDGES:
            if scs[i] < 0.3 or scs[j] < 0.3:
                continue
            ax_pose.plot([kps[i, 0], kps[j, 0]], [kps[i, 1], kps[j, 1]],
                         "-", color=pose.EDGE_COLORS.get((i, j), pose.CENTER_COLOR),
                         linewidth=2, alpha=0.8)
        for idx, (x, y) in enumerate(kps):
            if scs[idx] >= 0.3:
                ax_pose.plot(x, y, "o", color="white", markersize=4, zorder=5)
                ax_pose.plot(x, y, "o", color=pose.CENTER_COLOR, markersize=2, zorder=6)

    # ⑥ VLM 보고서
    ax_vlm = fig.add_subplot(gs[1, 2])
    ax_vlm.axis("off")
    ax_vlm.set_title("⑥ VLM Scene Report", fontsize=12, fontweight="bold")
    report_text = results.get("vlm_report", "No report generated.")
    # 긴 텍스트는 잘라서 표시
    if len(report_text) > 600:
        report_text = report_text[:600] + "..."
    ax_vlm.text(
        0.02, 0.95, report_text,
        transform=ax_vlm.transAxes,
        fontsize=8, verticalalignment="top",
        wrap=True,
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8),
    )

    fig.suptitle(
        "AI Scene Analyst — Foundation Model Pipeline",
        fontsize=15, fontweight="bold", y=1.01,
    )

    save_figure(fig, save_name)

    if show:
        plt.show()

    return fig


# ── 보고서 텍스트 저장 ─────────────────────────────────────────────────────────
def save_report(results: dict, save_name: str = "pipeline_report.txt") -> Path:
    det_result = results["detection"]
    depth_per_obj = results["depth_per_obj"]
    pose_results = results["pose"]
    vlm_report = results.get("vlm_report", "")

    out_path = OUTPUTS_DIR / save_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== AI Scene Analyst — Full Report ===\n\n")

        f.write("--- Detected Objects ---\n")
        for label, score in zip(det_result["labels"], det_result["scores"].tolist()):
            f.write(f"  {label}  (score={score:.3f})\n")

        f.write("\n--- Depth Ranking (nearest to farthest) ---\n")
        for obj in depth_per_obj:
            f.write(f"  {obj['rank']}. {obj['label']}  mean_depth={obj['mean_depth']:.4f}\n")

        if pose_results:
            f.write(f"\n--- Pose Analysis ({len(pose_results)} person) ---\n")
            for i, res in enumerate(pose_results):
                visible = int((res["scores"] >= 0.3).sum())
                f.write(f"  Person {i+1}: {visible}/17 keypoints\n")

        f.write("\n--- VLM Scene Report ---\n")
        f.write(vlm_report + "\n")

    print(f"[saved] {out_path}")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Scene Analyst — Full Foundation Model Pipeline (기말 프로젝트)"
    )
    parser.add_argument("--image", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                        help="탐지할 객체 프롬프트 (점으로 구분)")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-vlm-tokens", type=int, default=512)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = get_device()
    image = load_image(args.image)
    stem = Path(args.image).stem

    results = run_pipeline(
        image,
        detection_prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_vlm_tokens=args.max_vlm_tokens,
        device=device,
    )

    visualize_all(image, results,
                  save_name=f"{stem}_pipeline.png",
                  show=not args.no_show)

    save_report(results, save_name=f"{stem}_report.txt")

    print(f"\nAll results saved to: {OUTPUTS_DIR.resolve()}")
