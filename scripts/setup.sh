#!/usr/bin/env bash
# OpenEdit — Linux / macOS setup (Reframe gsplat: CUDA + torch 2.4 — see README)
# Usage: bash scripts/setup.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v git >/dev/null 2>&1; then
  echo "git required (clone apple/ml-sharp for Reframe)." >&2
  exit 1
fi

SHARP_DIR="third_party/ml-sharp"
if [ ! -d "$SHARP_DIR/.git" ]; then
  mkdir -p third_party
  echo "Cloning apple/ml-sharp into $SHARP_DIR ..."
  git clone --depth 1 https://github.com/apple/ml-sharp.git "$SHARP_DIR"
fi

PYTHON="${PYTHON:-python3.10}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.10 not found. Set PYTHON=python3.10 or install 3.10." >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install gradio==6.18.0 diffusers==0.36.0 transformers==5.5.0 accelerate==1.13.0 \
  opencv-python==4.11.0.86 timm==1.0.26 simple-lama-inpainting==0.1.1 qwen-vl-utils==0.0.14 \
  "imageio[ffmpeg]" "pillow>=10.0.0,<11.0.0"
pip install -e third_party/ml-sharp

echo ""
echo "Done. Activate: source .venv/bin/activate"
echo "Install PyTorch (+ gsplat on CUDA) per https://pytorch.org if not already present."
echo "Run: python demo.py"
