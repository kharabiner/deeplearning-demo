"""
task_vlm.py — Visual Language Model (VQA / Captioning)
모델: Qwen2-VL-2B-Instruct (Qwen/Qwen2-VL-2B-Instruct)

사용법:
    # 기본 이미지 설명
    python task_vlm.py --image sample.jpg

    # 커스텀 질문
    python task_vlm.py --image sample.jpg --question "이 이미지에서 이상한 점은?"

    # 여러 질문 연속
    python task_vlm.py --image sample.jpg --qa

동작:
    이미지 + 텍스트 질문 → 자연어 답변 생성
"""

import argparse
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from common import get_device, get_dtype, load_image

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# 기본 질문 목록 (--qa 모드에서 사용)
DEFAULT_QUESTIONS = [
    "Describe this image in detail.",
    "What objects can you identify in this image?",
    "What is the main subject of this image?",
    "Is there anything unusual or interesting in this image?",
    "What is the spatial relationship between the main objects?",
]


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_model(device: str):
    """
    Qwen2-VL processor + model 로드.

    Note:
        - CUDA: float16 사용 (메모리 절약)
        - MPS(M2): float32 강제 (MPS float16 불안정)
        - CPU: float32
    """
    print(f"[vlm] Loading model: {MODEL_ID}")
    dtype = get_dtype(device)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    # device_map은 CUDA에서만 사용 (MPS/CPU는 명시적 .to(device))
    # MPS는 float32만 지원하므로 float16 절대 사용 금지
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device != "cuda":
        model = model.to(device)

    model.eval()
    print(f"[vlm] Model ready on {device} ({dtype})")
    return processor, model


# ── 추론 ───────────────────────────────────────────────────────────────────────
def run(
    image: Union[Image.Image, str, Path],
    processor,
    model,
    question: str = "Describe this image in detail.",
    max_new_tokens: int = 512,
    device: Optional[str] = None,
) -> str:
    """
    이미지에 대한 질문에 답변 생성.

    Args:
        image: PIL Image 또는 이미지 파일 경로
        processor: Qwen2-VL processor
        model: Qwen2-VL model
        question: 이미지에 대한 질문 텍스트
        max_new_tokens: 생성할 최대 토큰 수
        device: 사용 device

    Returns:
        str: 모델의 답변 텍스트
    """
    if device is None:
        device = next(model.parameters()).device.type

    # 이미지를 PIL로 통일
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return answer.strip()


def run_multi(
    image: Union[Image.Image, str, Path],
    processor,
    model,
    questions: list[str],
    max_new_tokens: int = 512,
    device: Optional[str] = None,
) -> list[dict]:
    """
    여러 질문을 연속으로 처리.

    Returns:
        [{"question": str, "answer": str}, ...]
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    results = []
    for q in questions:
        answer = run(image, processor, model, q, max_new_tokens, device)
        results.append({"question": q, "answer": answer})

    return results


# ── 결과 출력 ──────────────────────────────────────────────────────────────────
def print_results(results: list[dict]) -> None:
    print(f"\n{'═'*60}")
    print("VLM Results")
    print(f"{'═'*60}")
    for i, r in enumerate(results, 1):
        print(f"\n[Q{i}] {r['question']}")
        print(f"{'─'*40}")
        print(f"[A{i}] {r['answer']}")
    print(f"\n{'═'*60}\n")


def save_results_text(results: list[dict], save_name: str = "vlm_result.txt") -> Path:
    """결과를 텍스트 파일로 저장."""
    from common import OUTPUTS_DIR
    out_path = OUTPUTS_DIR / save_name
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(results, 1):
            f.write(f"[Q{i}] {r['question']}\n")
            f.write(f"[A{i}] {r['answer']}\n\n")
    print(f"[saved] {out_path}")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Visual Language Model Q&A (Qwen2-VL-2B)")
    parser.add_argument("--image", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument(
        "--question", type=str,
        default="Describe this image in detail.",
        help="이미지에 대한 질문",
    )
    parser.add_argument(
        "--qa", action="store_true",
        help="기본 질문 5개를 연속으로 실행",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = get_device()
    print(f"[vlm] Device: {device}")

    image = load_image(args.image)
    print(f"[vlm] Image: {args.image}  size={image.size}")

    processor, model = load_model(device)

    stem = Path(args.image).stem

    if args.qa:
        questions = DEFAULT_QUESTIONS
    else:
        questions = [args.question]

    results = run_multi(image, processor, model, questions, args.max_tokens, device)
    print_results(results)
    save_results_text(results, save_name=f"{stem}_vlm.txt")
