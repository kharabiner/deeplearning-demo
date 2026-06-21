# OpenEdit — Python 3.10 + gsplat CUDA venv (.venv)
# Usage: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher (py) not found."
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git not found (required to clone apple/ml-sharp for Reframe)."
}

py -3.10 --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.10 required. Install: winget install Python.Python.3.10"
    exit 1
}

$sharpDir = "third_party\ml-sharp"
if (-not (Test-Path "$sharpDir\.git")) {
    New-Item -ItemType Directory -Force -Path "third_party" | Out-Null
    Write-Host "Cloning apple/ml-sharp into $sharpDir ..."
    git clone --depth 1 https://github.com/apple/ml-sharp.git $sharpDir
}

if (-not (Test-Path ".venv")) {
    py -3.10 -m venv .venv
}

$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip setuptools wheel
& $py -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install ninja
& $py -m pip install "gsplat==1.5.3+pt24cu124" --index-url https://docs.gsplat.studio/whl/pt24cu124/ --extra-index-url https://pypi.org/simple/
& $py -m pip install -r requirements.txt
& $py -m pip install gradio==6.18.0 diffusers==0.36.0 transformers==5.5.0 accelerate==1.13.0 `
    opencv-python==4.11.0.86 timm==1.0.26 simple-lama-inpainting==0.1.1 qwen-vl-utils==0.0.14 `
    "imageio[ffmpeg]"
& $py -m pip install -e third_party/ml-sharp
& $py -m pip install "pillow>=10.0.0,<11.0.0"

Write-Host ""
Write-Host "Done. Activate:"
Write-Host "  .venv\Scripts\activate"
Write-Host "Run:"
Write-Host "  python demo.py"
