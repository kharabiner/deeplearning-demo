# Deep Learning Assignment — Foundation Model Tasks

Hugging Face foundation model 기반 중간/기말 과제 코드입니다.

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
2. **가상환경 생성:** `python -m venv .venv`  
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

결과는 **`outputs/`** 폴더에 저장됩니다(저장소에는 `.gitignore`로 제외 — 로컬에서 실행 시 생성). `python <script> --help` 로 옵션을 확인할 수 있습니다.

---

## Brief explanation of each code (Input, Output, Example results)

아래는 중간 과제 요구사항(**각 코드에 대한 간단한 설명: 입력, 출력, 예시 결과**)에 맞춰 파일별로 정리한 것입니다.

### `common.py`

| | |
|--|--|
| **역할** | 모든 `task_*.py`가 공유하는 유틸리티: 디바이스(`cuda` / `mps` / `cpu`) 선택, 이미지 로드, 긴 변 기준 자동 resize, 결과 저장 경로(`outputs/`), MPS에서 float64 방지 등. |
| **입력 (Input)** | 직접 “과제 입력 이미지”를 받지 않음. 다른 스크립트가 `load_image(path)` 등으로 호출할 때 **이미지 파일 경로** 또는 PIL `Image`가 간접 입력. |
| **출력 (Output)** | 콘솔에 환경 요약(`device_info()`). 그림 저장은 각 `task_*.py`가 `save_figure` / `save_result`로 처리. |
| **예시** | `python common.py` → 현재 머신의 device·dtype·출력 폴더 경로 출력. |

---

### `task_detection.py` — Grounding DINO (open-vocabulary detection)

| | |
|--|--|
| **입력 (Input)** | **`--image`** 필수: RGB 이미지 파일 경로. **`--prompt`**: 찾을 객체를 영어로 마침표+공백으로 구분 (예: `person . laptop .`). 선택: `--box-threshold`, `--text-threshold`, `--no-show`. |
| **출력 (Output)** | **콘솔:** 탐지된 클래스·점수·박스 좌표. **파일:** 원본과 탐지 결과를 나란히 그린 PNG. |
| **예시 결과 (Example results)** | `python task_detection.py --image sample.jpg --prompt "person . laptop ."` 실행 후 **`outputs/sample_detection.png`** 생성. |

---

### `task_segmentation.py` — SAM 2 (mask generation)

| | |
|--|--|
| **입력 (Input)** | **`--image`** 필수. **`--mode`**: `auto`(이미지 전체 자동 마스크) 또는 `point`(이미지 중앙 한 점을 전경으로 세그멘테이션). **`--no-show`**: 창만 끄고 저장만. |
| **출력 (Output)** | **콘솔:** 마스크 개수, 면적, score. **파일:** 마스크를 색으로 겹친 시각화 PNG. |
| **예시 결과 (Example results)** | `python task_segmentation.py --image sample.jpg --mode auto` → **`outputs/sample_seg_auto.png`**. ` --mode point` → **`outputs/sample_seg_point.png`**. |

---

### `task_vlm.py` — Qwen2-VL-2B (image Q&A)

| | |
|--|--|
| **입력 (Input)** | **`--image`** 필수. **`--question`**: 이미지에 대한 한 줄 질문(영어 권장). **`--qa`**: 기본 질문 여러 개를 연속 실행. **`--max-tokens`**: 생성 최대 토큰. |
| **출력 (Output)** | **콘솔:** 질문·답변 텍스트. **파일:** (1) Q&A 전체 텍스트 **`.txt`**, (2) 왼쪽 입력 이미지·오른쪽 Q&A 패널 **`.png`**. |
| **예시 결과 (Example results)** | `python task_vlm.py --image sample.jpg --question "What is in this image?"` 실행 후 **`outputs/sample_vlm.txt`**, **`outputs/sample_vlm.png`**. |

---

### `task_depth.py` — Depth Anything V2 Small (monocular depth)

| | |
|--|--|
| **입력 (Input)** | **`--image`** 필수. 선택: **`--colormap`**(기본 `plasma`), **`--no-show`**. |
| **출력 (Output)** | **콘솔:** 깊이 맵 통계 등. **파일:** 왼쪽 원본·오른쪽 깊이 맵(컬러바; 값이 클수록 모델 기준 “가까움” 쪽). |
| **예시 결과 (Example results)** | `python task_depth.py --image sample.jpg` → **`outputs/sample_depth.png`**. |

---

### `task_pose.py` — ViTPose (human keypoints)

| | |
|--|--|
| **입력 (Input)** | **`--image`** 필수(사람이 보이는 장면). 선택: **`--score-threshold`**(낮은 신뢰도 관절 숨김, 기본 0.3), **`--no-show`**. BBox를 넘기지 않으면 **이미지 전체를 한 명**으로 가정. |
| **출력 (Output)** | **콘솔:** 사람별 보이는 키포인트 수. **파일:** 원본 + 스켈레톤 오버레이 PNG. |
| **예시 결과 (Example results)** | `python task_pose.py --image sample.jpg --score-threshold 0.6` → **`outputs/sample_pose.png`**. |

---

### `pipeline_final.py` (기말용 스켈레톤)

| | |
|--|--|
| **입력 (Input)** | **`--image`** 필수. 탐지용 **`--prompt`**, **`--box-threshold`**, **`--text-threshold`**, VLM용 **`--max-vlm-tokens`**, **`--no-show`**. |
| **출력 (Output)** | 탐지 → 세그멘테이션 → VLM 등 단계별 결과를 `outputs/`에 저장(실행 시 로그에 파일명 출력). |
| **예시** | `python pipeline_final.py --image sample.jpg` — 자세한 옵션은 `python pipeline_final.py --help`. 중간 과제 핵심 설명은 각 `task_*.py` 위 섹션을 사용. |

---

### 요약 표 (스크립트 ↔ 예시 출력 파일, `sample.jpg` 기준)

| Script | Example command | Example output files under `outputs/` |
|--------|-----------------|----------------------------------------|
| `task_detection.py` | `--image sample.jpg --prompt "person ."` | `sample_detection.png` |
| `task_segmentation.py` | `--image sample.jpg --mode auto` | `sample_seg_auto.png` |
| `task_segmentation.py` | `--image sample.jpg --mode point` | `sample_seg_point.png` |
| `task_vlm.py` | `--image sample.jpg --question "Describe this image."` | `sample_vlm.txt`, `sample_vlm.png` |
| `task_depth.py` | `--image sample.jpg` | `sample_depth.png` |
| `task_pose.py` | `--image sample.jpg` | `sample_pose.png` |

**“Three codes of foundation models”:** 서로 다른 foundation model을 쓰는 **`task_*.py`가 5개**이므로 최소 3개 요건을 충족합니다.

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
