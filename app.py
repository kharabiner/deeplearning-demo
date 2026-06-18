"""
app.py — OpenEdit 실행 진입점 (Gradio)

iOS 27 사진 편집 3기능을 오픈 파운데이션 모델로 재현:
  - Clean Up : 브러시/SAM2 세그 + (선택) Grounding DINO 텍스트 검출 → SD2 인페인팅
  - Expand   : 프레임 축소 → 바깥 영역 SD2 아웃페인팅(확장)
  - Reframe  : SHARP(단일이미지→3D Gaussian)로 시점 변경. LDI 미리보기 + SD2 생성.

코드 구성:
  - 실행 진입점 : app.py (이 파일)
  - 프론트엔드  : ui.py
  - 기능별 코드 : features/clean_up.py, features/expand.py, features/reframe.py
  - 공용 글루   : features/shared.py
  - 모델별 코드 : task_*.py, reframe_*.py, splat_render.py, inpaint.py

VRAM 8GB(RTX 3070 Ti) 대응 / device: CUDA·MPS·CPU.

실행:
    python app.py
    python app.py --share
"""

from __future__ import annotations

import argparse

try:  # 아이폰 사진 등 AVIF 입력 지원 (선택적)
    import pillow_avif  # noqa: F401
except Exception:
    pass

from features.shared import DEVICE
from ui import build_ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEdit Gradio app")
    parser.add_argument("--share", action="store_true", help="외부 공유 링크 생성")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print(f"[app] device = {DEVICE} · OpenEdit (Clean Up · Expand · Reframe)")
    port = args.port if args.port != 7860 else None
    build_ui().launch(share=args.share, server_port=port)
