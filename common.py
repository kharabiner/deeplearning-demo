"""
common.py — 공통 유틸리티
- device 자동 감지 (CUDA / MPS / CPU)
- MPS float64 방지 (MPS는 float64 미지원)
- 이미지 로드 / 안전한 resize / 저장
- 시각화 저장 헬퍼

강의 요구사항:
  - CPU / MPS(M2 Air) / CUDA(RTX3070) 모두 동작
  - clone 후 바로 실행 가능
  - 이미지 입력 시 에러 없도록 resize 처리
"""

import os
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch
from PIL import Image

# MPS는 float64 연산 미지원 → 프로세스 전역에서 float32 기본값 강제
# (torch.tensor(), torch.zeros() 등이 float64를 생성하는 것을 방지)
torch.set_default_dtype(torch.float32)


# ── 출력 폴더 ──────────────────────────────────────────────────────────────────
OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# 이미지 최대 크기 (가장 긴 변 기준 픽셀)
# 너무 큰 이미지는 OOM 또는 처리 시간 초과의 원인
MAX_IMAGE_SIZE = 1333


# ── Device ─────────────────────────────────────────────────────────────────────
def get_device() -> str:
    """CUDA(RTX) > MPS(M2) > CPU 순으로 가능한 device 반환."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    """
    float16은 CUDA에서만 안정적.
    MPS(M2)는 float16 일부 ops 미지원, float64 전혀 미지원 → float32 사용.
    CPU도 float32 사용.
    """
    return torch.float16 if device == "cuda" else torch.float32


def safe_to_float32(tensor: torch.Tensor) -> torch.Tensor:
    """
    float64 텐서를 float32로 변환 (MPS 에러 방지).
    float64가 아니면 그대로 반환.
    """
    if tensor.dtype == torch.float64:
        return tensor.to(torch.float32)
    return tensor


def device_info() -> str:
    """현재 device 정보 문자열 반환."""
    device = get_device()
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        return f"CUDA — {name} ({mem} MB VRAM)"
    if device == "mps":
        return "MPS — Apple Silicon (M-series, float32 only)"
    return "CPU — (float32, slow but always works)"


# ── 이미지 로드 & Resize ────────────────────────────────────────────────────────
def load_image(path: Union[str, Path]) -> Image.Image:
    """
    이미지를 RGB PIL.Image로 로드 후 안전한 크기로 자동 resize.
    모든 태스크 파일에서 이 함수를 통해 이미지를 로드해야 함.
    """
    image = Image.open(path).convert("RGB")
    return resize_if_needed(image)


def resize_if_needed(
    image: Image.Image,
    max_size: int = MAX_IMAGE_SIZE,
) -> Image.Image:
    """
    이미지의 가장 긴 변이 max_size를 초과하면 비율 유지하며 축소.

    이유:
      - 너무 큰 이미지는 모델 입력 시 OOM 또는 처리 시간 초과 발생
      - MPS는 큰 텐서 연산 시 float64로 업캐스팅 되는 경우가 있음
      - 1333px는 Grounding DINO / SAM2 권장 최대 입력 크기

    Returns:
      resize된 PIL Image (원본 크기 이하라면 그대로 반환)
    """
    W, H = image.size
    if max(W, H) <= max_size:
        return image

    scale = max_size / max(W, H)
    new_W = int(W * scale)
    new_H = int(H * scale)
    resized = image.resize((new_W, new_H), Image.LANCZOS)
    print(f"[resize] {W}x{H} → {new_W}x{new_H} (max_size={max_size})")
    return resized


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """PIL → uint8 NumPy (H, W, 3). float64 생성 없이 uint8 유지."""
    return np.asarray(image, dtype=np.uint8)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """uint8 NumPy (H, W, 3) → PIL."""
    return Image.fromarray(array.astype(np.uint8))


def numpy_to_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    """
    NumPy → torch.Tensor with float32 강제.
    numpy 기본 dtype이 float64인 경우 MPS에서 에러 발생 → float32로 변환.
    """
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    return tensor.to(device)


# ── 결과 저장 ──────────────────────────────────────────────────────────────────
def save_result(image: Union[Image.Image, np.ndarray], filename: str) -> Path:
    """
    결과 이미지를 outputs/ 폴더에 저장.
    Returns: 저장된 파일 경로
    """
    if isinstance(image, np.ndarray):
        image = numpy_to_pil(image)
    out_path = OUTPUTS_DIR / filename
    image.save(out_path)
    print(f"[saved] {out_path}")
    return out_path


def save_figure(fig, filename: str) -> Path:
    """matplotlib Figure를 outputs/ 폴더에 저장."""
    out_path = OUTPUTS_DIR / filename
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[saved] {out_path}")
    return out_path


# ── 간단한 이미지 그리드 시각화 ────────────────────────────────────────────────
def show_images(images: list, titles: list = None, cols: int = 2):
    """
    여러 이미지를 matplotlib 그리드로 표시.
    images: PIL or numpy 리스트
    """
    import matplotlib.pyplot as plt

    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for i, img in enumerate(images):
        if isinstance(img, Image.Image):
            img = pil_to_numpy(img)
        axes[i].imshow(img)
        axes[i].axis("off")
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=12)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig


# ── GPU 메모리 해제 ────────────────────────────────────────────────────────────
def free_memory(device: str) -> None:
    """GPU / MPS 캐시 해제 (모델 간 전환 시 사용)."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# ── 진입점 — 환경 확인 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Environment Check ===")
    device = get_device()
    dtype = get_dtype(device)
    print(f"Device  : {device_info()}")
    print(f"dtype   : {dtype}")
    print(f"Outputs : {OUTPUTS_DIR.resolve()}")
    print(f"PyTorch : {torch.__version__}")
    print(f"Default dtype: {torch.get_default_dtype()} (float64 disabled)")

    # float64 방지 확인
    t = torch.tensor([1.0, 2.0])
    assert t.dtype == torch.float32, "float32 default not set!"
    print("float32 default: OK")

    print("========================")
