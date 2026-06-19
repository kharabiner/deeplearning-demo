# Deep Learning Assignment — Foundation Model Tasks

Hugging Face foundation model 기반 중간/기말 과제 코드입니다.

**기말 데모:** **OpenEdit** — iOS 27 사진편집 3기능(Clean Up · Expand · Reframe) 통합 Gradio 앱. 아래 [OpenEdit (기말 데모)](#openedit-기말-데모) 섹션 참고.

---

## Repository

```cmd
git clone https://github.com/kharabiner/deeplearning-demo.git
cd deeplearning-demo
```

---

## Installation guide

**Python 3.10 필수** (Windows에서 gsplat CUDA 휠 + torch 2.4.1+cu124 조합). Python 3.11 `.venv`는 Reframe gsplat 미지원( torch splat 폴백만 가능).

### Windows (권장 — 한 번에 설치)

```cmd
git clone https://github.com/kharabiner/deeplearning-demo.git
cd deeplearning-demo
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.venv\Scripts\activate
python app.py
```

`scripts\setup.ps1` 이 `.venv` 생성 후 **torch 2.4.1+cu124 · gsplat · diffusers 0.36.0 · ml-sharp** 등을 설치합니다.

### 수동 설치 (Linux / macOS 또는 커스텀)

```cmd
py -3.10 -m venv .venv
.venv\Scripts\activate.bat          # Windows
# source .venv/bin/activate         # Linux / macOS

.venv\Scripts\python.exe -m pip install -r requirements.txt
# NVIDIA GPU: https://pytorch.org 에서 CUDA/MPS 빌드 설치 후 gsplat·ml-sharp 추가
pip install -e third_party/ml-sharp
```

가중치(SHARP·HF 모델)는 최초 실행 시 자동 다운로드(토큰 불필요).

### Device

- **CUDA (NVIDIA, Windows)**: 전 기능 권장 · Reframe **gsplat** 렌더
- **MPS (Apple Silicon)**: Clean Up / Expand 동작 · Reframe gsplat 없음
- **CPU**: Clean Up / Expand만 권장(느림) · Reframe 비권장

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
- **Expand (확장)**: LaMa 미리보기 배경 → 슬라이더로 프레임 축소·실시간 미리보기 → [완료] **DreamShaper** 아웃페인팅.
- **Reframe (시점 변경)**: **SHARP + gsplat**. 슬라이더 수치(좌우 -16~+16, 상하 -5~+5, **1당 3°**) · 미리보기 블러 · [완료] gsplat + **DreamShaper** 바깥 생성.

### 사용 파운데이션 모델


| 모델                   | HF ID / 출처                                                                     | 역할                      |
| -------------------- | ------------------------------------------------------------------------------ | ----------------------- |
| Apple SHARP          | `apple/ml-sharp` + 공개 체크포인트                                                    | Reframe 3D Gaussian      |
| gsplat               | CUDA wheel (torch 2.4.1)                                                       | Reframe splat 렌더        |
| SAM2                 | `facebook/sam2-hiera-base-plus`                                                | Clean Up 세그멘테이션         |
| Grounding DINO       | `IDEA-Research/grounding-dino-base`                                            | Clean Up 텍스트 검색         |
| Qwen2-VL-2B          | `Qwen/Qwen2-VL-2B-Instruct`                                                    | SD 인페인팅 프롬프트 (VLM)      |
| DreamShaper Inpaint  | `Lykon/dreamshaper-8-inpainting`                                               | Clean Up · Expand · Reframe [완료] |
| LaMa                 | `big-lama` (simple-lama-inpainting)                                            | Expand 미리보기              |

> Windows `.venv`: **Python 3.10** · torch **2.4.1+cu124** · diffusers **0.36.0** (`scripts\setup.ps1`).

### Device

- **CUDA (NVIDIA)**: 전 기능 권장 · Reframe gsplat
- **MPS**: Clean Up / Expand · Reframe gsplat 없음
- **CPU**: Clean Up / Expand만 권장

### 성능 팁

- **Reframe**: yaw×pitch 격자 전체 프리렌더 → GPU 수 분 가능. 상수: `features/reframe.py`.
- **인페인팅**: DreamShaper SD1.5 (Clean Up · Expand · Reframe [완료]), LaMa (Expand 미리보기).
- **VRAM**: 모델 순차 load/unload · [완료] 시 SD 로드 + 수십 초.

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

결과는 `**outputs/**` 폴더에 저장됩니다(저장소에는 `.gitignore`로 제외 — 로컬에서 실행 시 생성). `python <script> --help` 로 옵션을 확인할 수 있습니다.

---

## Brief explanation of each code (Input, Output, Example results)

아래는 중간 과제 요구사항(**각 코드에 대한 간단한 설명: 입력, 출력, 예시 결과**)에 맞춰 파일별로 정리한 것입니다.

### `common.py`


|                 |                                                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **역할**          | 모든 `task_*.py`가 공유하는 유틸리티: 디바이스(`cuda` / `mps` / `cpu`) 선택, 이미지 로드, 긴 변 기준 자동 resize, 결과 저장 경로(`outputs/`), MPS에서 float64 방지 등. |
| **입력 (Input)**  | 직접 “과제 입력 이미지”를 받지 않음. 다른 스크립트가 `load_image(path)` 등으로 호출할 때 **이미지 파일 경로** 또는 PIL `Image`가 간접 입력.                               |
| **출력 (Output)** | 콘솔에 환경 요약(`device_info()`). 그림 저장은 각 `task_*.py`가 `save_figure` / `save_result`로 처리.                                            |
| **예시**          | `python common.py` → 현재 머신의 device·dtype·출력 폴더 경로 출력.                                                                           |


---

### `task_detection_groundingdino.py` — Grounding DINO (open-vocabulary detection)


|                             |                                                                                                                                                           |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **입력 (Input)**              | `**--image`** 필수: RGB 이미지 파일 경로. `**--prompt**`: 찾을 객체를 영어로 마침표+공백으로 구분 (예: `person . laptop .`). 선택: `--box-threshold`, `--text-threshold`, `--no-show`. |
| **출력 (Output)**             | **콘솔:** 탐지된 클래스·점수·박스 좌표. **파일:** 원본과 탐지 결과를 나란히 그린 PNG.                                                                                                  |
| **예시 결과 (Example results)** | `python task_detection_groundingdino.py --image sample.jpg --prompt "person . laptop ."` 실행 후 `**outputs/sample_detection.png`** 생성.                      |


---

### `task_segmentation_sam2.py` — SAM 2 (mask generation)


|                             |                                                                                                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **입력 (Input)**              | `**--image`** 필수. `**--mode**`: `auto`(이미지 전체 자동 마스크) 또는 `point`(이미지 중앙 한 점을 전경으로 세그멘테이션). `**--no-show**`: 창만 끄고 저장만.                                      |
| **출력 (Output)**             | **콘솔:** 마스크 개수, 면적, score. **파일:** 마스크를 색으로 겹친 시각화 PNG.                                                                                                     |
| **예시 결과 (Example results)** | `python task_segmentation_sam2.py --image sample.jpg --mode auto` → `**outputs/sample_seg_auto.png`**. `--mode point` → `**outputs/sample_seg_point.png**`. |


---

### `task_vlm_qwen2vl.py` — Qwen2-VL-2B (image Q&A)


|                             |                                                                                                                                                      |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **입력 (Input)**              | `**--image`** 필수. `**--question**`: 이미지에 대한 한 줄 질문(영어 권장). `**--qa**`: 기본 질문 여러 개를 연속 실행. `**--max-tokens**`: 생성 최대 토큰.                              |
| **출력 (Output)**             | **콘솔:** 질문·답변 텍스트. **파일:** (1) Q&A 전체 텍스트 `**.txt`**, (2) 왼쪽 입력 이미지·오른쪽 Q&A 패널 `**.png**`.                                                           |
| **예시 결과 (Example results)** | `python task_vlm_qwen2vl.py --image sample.jpg --question "What is in this image?"` 실행 후 `**outputs/sample_vlm.txt`**, `**outputs/sample_vlm.png**`. |


---

### `task_depth_depthanythingv2.py` — Depth Anything V2 Small (monocular depth)


|                             |                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| **입력 (Input)**              | `**--image`** 필수. 선택: `**--colormap**`(기본 `plasma`), `**--no-show**`.                       |
| **출력 (Output)**             | **콘솔:** 깊이 맵 통계 등. **파일:** 왼쪽 원본·오른쪽 깊이 맵(컬러바; 값이 클수록 모델 기준 “가까움” 쪽).                       |
| **예시 결과 (Example results)** | `python task_depth_depthanythingv2.py --image sample.jpg` → `**outputs/sample_depth.png`**. |


---

### `task_pose_vitpose.py` — ViTPose (human keypoints)


|                             |                                                                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **입력 (Input)**              | `**--image`** 필수(사람이 보이는 장면). 선택: `**--score-threshold**`(낮은 신뢰도 관절 숨김, 기본 0.3), `**--no-show**`. BBox를 넘기지 않으면 **이미지 전체를 한 명**으로 가정. |
| **출력 (Output)**             | **콘솔:** 사람별 보이는 키포인트 수. **파일:** 원본 + 스켈레톤 오버레이 PNG.                                                                                   |
| **예시 결과 (Example results)** | `python task_pose_vitpose.py --image sample.jpg --score-threshold 0.6` → `**outputs/sample_pose.png`**.                               |


---

### `pipeline_final.py` (미구현)

기말 통합 데모는 `**app.py` (OpenEdit)** 를 사용하세요. 개별 `task_*.py` 테스트는 아래 Run 섹션 참고.

---

### 요약 표 (스크립트 ↔ 예시 출력 파일, `sample.jpg` 기준)


| Script                            | Example command                                        | Example output files under `outputs/` |
| --------------------------------- | ------------------------------------------------------ | ------------------------------------- |
| `task_detection_groundingdino.py` | `--image sample.jpg --prompt "person ."`               | `sample_detection.png`                |
| `task_segmentation_sam2.py`       | `--image sample.jpg --mode auto`                       | `sample_seg_auto.png`                 |
| `task_segmentation_sam2.py`       | `--image sample.jpg --mode point`                      | `sample_seg_point.png`                |
| `task_vlm_qwen2vl.py`             | `--image sample.jpg --question "Describe this image."` | `sample_vlm.txt`, `sample_vlm.png`    |
| `task_depth_depthanythingv2.py`   | `--image sample.jpg`                                   | `sample_depth.png`                    |
| `task_pose_vitpose.py`            | `--image sample.jpg`                                   | `sample_pose.png`                     |


**“Three codes of foundation models”:** 서로 다른 foundation model을 쓰는 `**task_*.py`가 5개**이므로 최소 3개 요건을 충족합니다.

---

## 파일 구조

```
deeplearning/
  app.py                 # 실행 진입점
  ui.py                  # Gradio UI
  features/
    shared.py            # VLM 캡션 · 인페인팅 합성
    clean_up.py          # Clean Up
    expand.py            # Expand (LaMa + DreamShaper)
    reframe.py           # Reframe (gsplat + DreamShaper)
  reframe_yaw.py         # yaw×pitch 프리렌더 그리드
  sharp_render.py        # SHARP + gsplat 렌더
  splat_torch.py         # gsplat 불가 시 PyTorch 폴백
  inpaint.py             # 인페인팅 팩토리 (sd15 / lama / opencv)
  task_*.py              # 중간 과제 단독 스크립트
  task_inpaint_sd15.py   # DreamShaper SD1.5 inpaint
  scripts/setup.ps1      # Windows: Python 3.10 + .venv 일괄 설치
  requirements.txt
  third_party/ml-sharp   # SHARP (setup.ps1 에서 editable install)
  sample.jpg
  outputs/               # 결과 (git 제외)
```

---

## 사용 모델


| 태스크          | 모델 이름                   | Hugging Face ID                             | 크기(대략) |
| ------------ | ----------------------- | ------------------------------------------- | ------ |
| Detection    | Grounding DINO (base)   | `IDEA-Research/grounding-dino-base`         | ~811MB |
| Segmentation | SAM 2 Hiera base+       | `facebook/sam2-hiera-base-plus`             | ~320MB |
| VLM          | Qwen2-VL-2B-Instruct    | `Qwen/Qwen2-VL-2B-Instruct`                 | ~4GB   |
| Depth        | Depth Anything V2 Small | `depth-anything/Depth-Anything-V2-Small-hf` | ~97MB  |
| Pose         | ViTPose base (simple)   | `usyd-community/vitpose-base-simple`        | ~330MB |


모델은 첫 실행 시 Hugging Face Hub에서 자동 다운로드됩니다.