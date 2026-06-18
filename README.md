# Deep Learning Assignment — Foundation Model Tasks

Hugging Face foundation model 기반 중간/기말 과제 코드입니다.

**기말 데모:** **OpenEdit** — iOS 27 사진편집 3기능(Clean Up · Expand · Reframe) 통합 Gradio 앱. 아래 [OpenEdit (기말 데모)](#openedit-기말-데모) 섹션 참고.

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

5. **SHARP 설치 (Reframe용, 선택이지만 권장)**

```cmd
# Windows
powershell -ExecutionPolicy Bypass -File scripts\setup_sharp.ps1

# Linux / macOS
bash scripts/setup_sharp.sh
```

가중치는 최초 Reframe 실행 시 공개 URL에서 자동 다운로드됩니다(토큰 불필요).

6. **GPU (선택)**  
   - NVIDIA: CUDA용 PyTorch가 설치된 환경이면 자동으로 `cuda` 사용  
   - Apple Silicon: `mps` 사용 (float32)  
   - 그 외: `cpu`

---

## OpenEdit (기말 데모)

iOS 27 **Spatial Reframing**을 오픈 파운데이션 모델로 재현한 통합 데모입니다.

### 실행

```cmd
python app.py
python app.py --share    # 외부 공유 링크
```

브라우저에서 사진 업로드 → **Clean Up / Expand / Reframe** 버튼 → 슬라이더·브러시로 미리보기 → **완료**로 인페인팅.

#### 기능 설명

- **Clean Up (지우기)**: 브러시로 문질러서 지우고 싶은 객체를 선택하면 SAM2로 자동 세그멘테이션 후 AI가 제거. 텍스트 검색으로도 객체 선택 가능.
- **Expand (확장)**: 사진 프레임을 축소하여 바깥 영역을 AI로 자연스럽게 확장. 슬라이더로 실시간 미리보기.
- **Reframe (시점 변경)**: 마치 사진을 찍기 전에 카메라 각도를 바꾸는 것처럼 시점 변경. Reframe 버튼을 누르면 SHARP 3D 분석 + LDI 전경/사람 레이어 분리(로딩). 이후 슬라이더로 각도 조정 시 실시간 미리보기에서 사람·사물은 깔끔히 회전하고 바깥쪽은 블러로 표시. [완료] 버튼으로 SHARP 고품질 렌더 + 바깥 영역을 SD2로 생성.



### 사용 파운데이션 모델

| 모델 | HF ID / 출처 | 역할 |
|------|----------------|------|
| Depth Anything V2 | `depth-anything/Depth-Anything-V2-Small-hf` | 뎁스 → 3D 워핑 (Reframe) |
| SAM2 | `facebook/sam2-hiera-base-plus` | Clean Up 세그멘테이션 |
| Grounding DINO | `IDEA-Research/grounding-dino-base` | Clean Up 텍스트 검색 |
| Qwen2-VL-2B | `Qwen/Qwen2-VL-2B-Instruct` | SD 인페인팅 프롬프트 (VLM) |
| LaMa / SD1.5 Inpaint | `simple-lama-inpainting` / `stable-diffusion-v1-5/stable-diffusion-inpainting` | 인페인팅 (객체 제거 및 영역 생성) |
| Apple SHARP | `apple/ml-sharp` + 공개 체크포인트 | Reframe 3D Gaussian 렌더링 |

### Device

- **CUDA** (NVIDIA): 권장, 전 기능 고성능
- **MPS** (Apple Silicon): Clean Up/Expand/Reframe 동작 (float32)
- **CPU**: Clean Up/Expand만 동작(느림). Reframe은 SHARP 미설치 시 depth 폴백

### 성능 팁

- **Reframe 성능**: SHARP 설치 시 최고 품질. 미설치 시 depth 워핑 사용 (빠르지만 품질 낮음).
- **인페인팅 백엔드**: `lama` (빠름, 단순한 채움), `sd2` (VLM 프롬프트 사용, 고품질), `opencv` (가장 빠름, 기본 채움).
- **각도 조정**: Reframe에서 소폭 각도 변경(±12° 이내) 권장. 큰 각도는 왜곡 발생 가능.

---

## Run (quick test)

저장소에 포함된 `sample.jpg`로 바로 실행할 수 있습니다.

```cmd
python common.py
python task_detection_groundingdino.py --image sample.jpg --prompt "person . laptop . bottle ."
python task_segmentation_sam2.py --image sample.jpg --mode auto
python task_vlm_qwen2vl.py --image sample.jpg --question "Describe this image."
python task_depth_depthanythingv2.py --image sample.jpg
python task_pose_vitpose.py --image sample.jpg --score-threshold 0.6
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

### `task_detection_groundingdino.py` — Grounding DINO (open-vocabulary detection)

| | |
|--|--|
| **입력 (Input)** | **`--image`** 필수: RGB 이미지 파일 경로. **`--prompt`**: 찾을 객체를 영어로 마침표+공백으로 구분 (예: `person . laptop .`). 선택: `--box-threshold`, `--text-threshold`, `--no-show`. |
| **출력 (Output)** | **콘솔:** 탐지된 클래스·점수·박스 좌표. **파일:** 원본과 탐지 결과를 나란히 그린 PNG. |
| **예시 결과 (Example results)** | `python task_detection_groundingdino.py --image sample.jpg --prompt "person . laptop ."` 실행 후 **`outputs/sample_detection.png`** 생성. |

---

### `task_segmentation_sam2.py` — SAM 2 (mask generation)

| | |
|--|--|
| **입력 (Input)** | **`--image`** 필수. **`--mode`**: `auto`(이미지 전체 자동 마스크) 또는 `point`(이미지 중앙 한 점을 전경으로 세그멘테이션). **`--no-show`**: 창만 끄고 저장만. |
| **출력 (Output)** | **콘솔:** 마스크 개수, 면적, score. **파일:** 마스크를 색으로 겹친 시각화 PNG. |
| **예시 결과 (Example results)** | `python task_segmentation_sam2.py --image sample.jpg --mode auto` → **`outputs/sample_seg_auto.png`**. ` --mode point` → **`outputs/sample_seg_point.png`**. |

---

### `task_vlm_qwen2vl.py` — Qwen2-VL-2B (image Q&A)

| | |
|--|--|
| **입력 (Input)** | **`--image`** 필수. **`--question`**: 이미지에 대한 한 줄 질문(영어 권장). **`--qa`**: 기본 질문 여러 개를 연속 실행. **`--max-tokens`**: 생성 최대 토큰. |
| **출력 (Output)** | **콘솔:** 질문·답변 텍스트. **파일:** (1) Q&A 전체 텍스트 **`.txt`**, (2) 왼쪽 입력 이미지·오른쪽 Q&A 패널 **`.png`**. |
| **예시 결과 (Example results)** | `python task_vlm_qwen2vl.py --image sample.jpg --question "What is in this image?"` 실행 후 **`outputs/sample_vlm.txt`**, **`outputs/sample_vlm.png`**. |

---

### `task_depth_depthanythingv2.py` — Depth Anything V2 Small (monocular depth)

| | |
|--|--|
| **입력 (Input)** | **`--image`** 필수. 선택: **`--colormap`**(기본 `plasma`), **`--no-show`**. |
| **출력 (Output)** | **콘솔:** 깊이 맵 통계 등. **파일:** 왼쪽 원본·오른쪽 깊이 맵(컬러바; 값이 클수록 모델 기준 “가까움” 쪽). |
| **예시 결과 (Example results)** | `python task_depth_depthanythingv2.py --image sample.jpg` → **`outputs/sample_depth.png`**. |

---

### `task_pose_vitpose.py` — ViTPose (human keypoints)

| | |
|--|--|
| **입력 (Input)** | **`--image`** 필수(사람이 보이는 장면). 선택: **`--score-threshold`**(낮은 신뢰도 관절 숨김, 기본 0.3), **`--no-show`**. BBox를 넘기지 않으면 **이미지 전체를 한 명**으로 가정. |
| **출력 (Output)** | **콘솔:** 사람별 보이는 키포인트 수. **파일:** 원본 + 스켈레톤 오버레이 PNG. |
| **예시 결과 (Example results)** | `python task_pose_vitpose.py --image sample.jpg --score-threshold 0.6` → **`outputs/sample_pose.png`**. |

---

### `pipeline_final.py` (미구현)

기말 통합 데모는 **`app.py` (OpenEdit)** 를 사용하세요. 개별 `task_*.py` 테스트는 아래 Run 섹션 참고.

---

### 요약 표 (스크립트 ↔ 예시 출력 파일, `sample.jpg` 기준)

| Script | Example command | Example output files under `outputs/` |
|--------|-----------------|----------------------------------------|
| `task_detection_groundingdino.py` | `--image sample.jpg --prompt "person ."` | `sample_detection.png` |
| `task_segmentation_sam2.py` | `--image sample.jpg --mode auto` | `sample_seg_auto.png` |
| `task_segmentation_sam2.py` | `--image sample.jpg --mode point` | `sample_seg_point.png` |
| `task_vlm_qwen2vl.py` | `--image sample.jpg --question "Describe this image."` | `sample_vlm.txt`, `sample_vlm.png` |
| `task_depth_depthanythingv2.py` | `--image sample.jpg` | `sample_depth.png` |
| `task_pose_vitpose.py` | `--image sample.jpg` | `sample_pose.png` |

**“Three codes of foundation models”:** 서로 다른 foundation model을 쓰는 **`task_*.py`가 5개**이므로 최소 3개 요건을 충족합니다.

---

## 파일 구조

```
deeplearning/
  app.py                 # 실행 진입점 (argparse + launch)
  ui.py                  # 프론트엔드 화면 구성 (Gradio Blocks)
  features/              # 기능별 코드
    __init__.py
    shared.py            #   공용 글루 (VLM 캡션·깊이·인페인팅 합성)
    clean_up.py          #   Clean Up (객체 지우기)
    expand.py            #   Expand (프레임 확장/아웃페인팅)
    reframe.py           #   Reframe (시점 변경, LDI 미리보기)
  requirements.txt       # 의존성
  scripts/setup_sharp.*  # SHARP 설치
  common.py              # 공통 유틸: device 감지, 이미지 로드/resize
  reframe_core.py        # depth 워핑 엔진
  reframe_sharp.py       # SHARP 렌더
  reframe_ldi.py         # LDI 2-레이어 (Expand·Reframe 미리보기)
  reframe_layers.py      # 사람 빌보드 레이어링
  splat_render.py        # 순수 PyTorch 3DGS 렌더러
  inpaint.py             # 인페인팅 팩토리 (sd2 / lama / opencv)
  task_detection_groundingdino.py    # Grounding DINO — open-vocab 탐지
  task_segmentation_sam2.py          # SAM2 — 픽셀 세그멘테이션
  task_vlm_qwen2vl.py                # Qwen2-VL-2B — 이미지 질의응답
  task_depth_depthanythingv2.py      # Depth Anything V2 — 깊이 추정
  task_nvs_sharp.py                  # Apple SHARP — 3D Gaussian 예측
  task_inpaint_sd.py                 # SD 인페인팅
  task_inpaint_lama.py               # LaMa 인페인팅
  task_pose_vitpose.py               # ViTPose — 관절 키포인트
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
