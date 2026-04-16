# Deep Learning Assignment — Foundation Model Tasks

Hugging Face foundation model 기반 중간/기말 과제 코드입니다.

**Midterm GitHub submission:** 이 저장소는 요구사항(README, 설치 가이드, 각 코드 설명, **3개 이상**의 foundation model 코드)을 충족합니다. foundation model 태스크 스크립트는 `task_*.py` 다섯 개입니다.

---

## Repository

```cmd
git clone https://github.com/kharabiner/deeplearning-demo.git
cd deeplearning-demo
```

(로컬 폴더 이름은 `deeplearning`으로 바꿔도 됩니다.)

---

## Installation guide

1. **Python** 3.10 이상 권장  
2. **가상환경 생성**

```cmd
python -m venv .venv
```

3. **가상환경 활성화**

```cmd
# Windows
.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

4. **의존성 설치**

```cmd
# Windows — 반드시 venv의 python으로 설치 (시스템 pip 혼동 방지)
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Linux / macOS
pip install -r requirements.txt
```

5. **GPU (선택)**  
   - NVIDIA: CUDA용 PyTorch가 설치된 환경이면 자동으로 `cuda` 사용  
   - Apple Silicon: `mps` 사용 (float32)  
   - 그 외: `cpu`

---

## Run (quick test)

저장소에 포함된 `sample.jpg`로 바로 실행할 수 있습니다.

```cmd
python common.py
python task_detection.py --image sample.jpg --prompt "person . laptop . bottle ."
python task_segmentation.py --image sample.jpg --mode auto
python task_vlm.py --image sample.jpg --question "Describe this image."
python task_depth.py --image sample.jpg
python task_pose.py --image sample.jpg --score-threshold 0.6
```

결과는 **`outputs/`** 에 PNG(또는 VLM은 TXT)로 저장됩니다. `python … --help` 로 각 스크립트 옵션을 확인할 수 있습니다.

---

## 각 코드 설명 (입력 / 출력 / 예시 결과)

| 스크립트 | Foundation model | 입력 (Input) | 출력 (Output) | 예시 결과 파일 (`outputs/`) |
|----------|------------------|----------------|----------------|-----------------------------|
| `task_detection.py` | Grounding DINO | 이미지 경로 `--image`, 텍스트 프롬프트 `--prompt` (예: `"person . cat ."`) | 바운딩 박스 + 라벨이 그려진 이미지 | `{이미지stem}_detection.png` |
| `task_segmentation.py` | SAM 2 | 이미지 `--image`, 모드 `--mode` (`auto` \| `point`) | 마스크 오버레이 시각화 | `{stem}_seg_auto.png` 또는 `{stem}_seg_point.png` |
| `task_vlm.py` | Qwen2-VL-2B | 이미지 `--image`, 질문 `--question` (또는 `--qa` 로 기본 질문 여러 개) | 질문·답변 텍스트 + 이미지+QA 패널 PNG | `{stem}_vlm.txt`, `{stem}_vlm.png` |
| `task_depth.py` | Depth Anything V2 Small | 이미지 `--image` | 원본 + 깊이 맵(컬러바) | `{stem}_depth.png` |
| `task_pose.py` | ViTPose | 이미지 `--image`, 선택 `--score-threshold` | 원본 + 스켈레톤 오버레이 | `{stem}_pose.png` |

공통: `common.py` — 디바이스 감지, 이미지 로드·resize, 결과 저장 헬퍼.  
`pipeline_final.py` — 기말용으로 위 태스크들을 한 이미지에 연결하는 스켈레톤.

**과제 문구 “Three codes of foundation models”:** 위 표의 **5개** `task_*.py`가 각각 서로 다른 Hugging Face foundation model을 사용하므로, “최소 3개” 요건을 초과 충족합니다.

---

## 파일 구조

```
deeplearning/
  common.py              # 공통 유틸: device 감지, 이미지 로드/resize
  task_detection.py      # Grounding DINO — open-vocab 탐지
  task_segmentation.py   # SAM2 — 픽셀 세그멘테이션
  task_vlm.py            # Qwen2-VL-2B — 이미지 질의응답
  task_depth.py          # Depth Anything V2 — 깊이 추정
  task_pose.py           # ViTPose — 관절 키포인트
  pipeline_final.py      # 기말용: 전체 파이프라인
  sample.jpg             # 샘플 입력 이미지
  outputs/               # 결과 저장 폴더 (git 제외)
```

---

## 사용 모델

| 태스크 | 모델 이름 | Hugging Face ID | 크기(대략) |
|--------|-----------|-----------------|------------|
| Detection | Grounding DINO (base) | `IDEA-Research/grounding-dino-base` | ~811MB |
| Segmentation | SAM 2 Hiera base+ | `facebook/sam2-hiera-base-plus` | ~320MB |
| VLM | Qwen2-VL-2B-Instruct | `Qwen/Qwen2-VL-2B-Instruct` | ~4GB |
| Depth | Depth Anything V2 Small | `depth-anything/Depth-Anything-V2-Small-hf` | ~97MB |
| Pose | ViTPose base (simple) | `usyd-community/vitpose-base-simple` | ~330MB |

모델은 첫 실행 시 Hugging Face Hub에서 자동 다운로드됩니다.
