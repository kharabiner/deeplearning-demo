# Deep Learning Assignment — Foundation Model Tasks

Hugging Face foundation model 기반 **중간 과제** 단독 스크립트 + **기말 통합 데모** 코드

## 기말 데모 — OpenEdit

iOS 27 사진편집 3기능(Clean Up · Expand · Reframe)**을 오픈 파운데이션 모델로 재현한 통합 Gradio 앱


| 요건 (기말 과제)         | 충족                                  |
| ------------------ | ----------------------------------- |
| 파운데이션 모델 3개 이상     | **6종** (아래 표) — VLM(Qwen2-VL) 포함    |
| HF 토큰·등록 없이 다운로드   | 전 모델 공개 ID, 최초 실행 시 자동 캐시           |
| fully reproducible | `git clone` → 설치 → `python demo.py` |
| CUDA / CPU / MPS   | `common.get_device()` 자동 분기         |
| README 설치·실행 안내    | 본 문서                                |


---

## Repository

```cmd
git clone https://github.com/kharabiner/deeplearning-demo.git
cd deeplearning-demo
```

---

## Installation

**Python 3.10 권장** (Windows에서 gsplat CUDA wheel + torch 2.4.1+cu124 조합). Python 3.11 `.venv`는 Reframe gsplat 미지원(torch splat 폴백만 가능).

### Windows (한 번에 설치)

```cmd
git clone https://github.com/kharabiner/deeplearning-demo.git
cd deeplearning-demo
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.venv\Scripts\activate
python demo.py
```

`scripts\setup.ps1`이 **apple/ml-sharp clone**(Reframe) → `.venv` 생성 → **torch 2.4.1+cu124 · gsplat · diffusers 0.36.0** 등을 설치합니다. `git`이 PATH에 있어야 합니다.

### Linux / macOS (한 번에 설치)

```bash
git clone https://github.com/kharabiner/deeplearning-demo.git
cd deeplearning-demo
bash scripts/setup.sh
source .venv/bin/activate
# NVIDIA GPU: https://pytorch.org 에서 CUDA torch 2.4 설치 후 (Reframe gsplat은 setup.ps1 README 참고)
python demo.py
```

`scripts/setup.sh`도 **ml-sharp를 자동 clone**합니다. PyTorch·gsplat(CUDA)은 플랫폼별로 별도 설치가 필요할 수 있습니다.

### SHARP (Reframe) — `third_party/ml-sharp`

레포에는 SHARP **소스 코드가 포함되지 않습니다** (`.gitignore`). 설치 스크립트가 없으면 수동 clone:

```bash
mkdir -p third_party
git clone --depth 1 https://github.com/apple/ml-sharp.git third_party/ml-sharp
pip install -e third_party/ml-sharp
```

SHARP **가중치**(`sharp_2572gikvuh.pt`)는 최초 Reframe 실행 시 Apple CDN에서 자동 다운로드됩니다.

### 수동 설치 (Linux / macOS)

```bash
python3.10 -m venv .venv
source .venv/bin/activate

mkdir -p third_party
git clone --depth 1 https://github.com/apple/ml-sharp.git third_party/ml-sharp

pip install --upgrade pip setuptools wheel
# NVIDIA GPU: https://pytorch.org 에서 CUDA 빌드 설치
pip install -r requirements.txt
pip install gradio==6.18.0 diffusers==0.36.0 transformers==5.5.0 accelerate==1.13.0 \
    opencv-python==4.11.0.86 timm==1.0.26 simple-lama-inpainting==0.1.1 qwen-vl-utils==0.0.14 \
    "imageio[ffmpeg]" "pillow>=10.0.0,<11.0.0"
pip install -e third_party/ml-sharp
# Reframe gsplat (CUDA, torch 2.4): setup.ps1 참고 — 없으면 PyTorch splat 폴백
```

가중치(SHARP·HF 모델)는 **최초 실행 시 자동 다운로드**(토큰 불필요). Hugging Face Hub 캐시에 저장되어 재실행 시 재사용됩니다.

### Device


| Device                  | Clean Up | Expand | Reframe                    |
| ----------------------- | -------- | ------ | -------------------------- |
| **CUDA (NVIDIA)**       | ✅        | ✅      | ✅ gsplat 권장                |
| **MPS (Apple Silicon)** | ✅        | ✅      | ⚠️ gsplat 없음 — Reframe 비권장 |
| **CPU**                 | ✅ (느림)   | ✅ (느림) | ❌ 비권장                      |


- **개발·테스트:** Windows + RTX 3070 Ti(8GB) 등에서 전 기능 동작. 8GB VRAM은 모델 **순차 load/unload**.

---

## Run — OpenEdit (기말 데모 · `demo.py`)

```cmd
python demo.py
python demo.py --share    # 외부 공유 링크 (Gradio)
python demo.py --port 7860
```

### 기능


| 기능           | 동작                                       | 미리보기                                            | [완료]                                 |
| ------------ | ---------------------------------------- | ----------------------------------------------- | ------------------------------------ |
| **Clean Up** | 브러시 획 → **SAM2** 객체 마스크                  | 마스크 오버레이                                        | **Qwen2-VL** 캡션 → **DreamShaper** 제거 |
| **Expand**   | 프레임 축소·바깥 여백 노출                          | **LaMa** 배경 + 슬라이더 줌                            | **DreamShaper** 아웃페인팅                |
| **Reframe**  | **SHARP** 3D Gaussian + **gsplat** 시점 이동 | 슬라이더(좌우 −16~~+16, 상하 −5~~+5, **1당 5°**) · 구멍 블러 | gsplat + **DreamShaper** 디오클루전 채움    |


### 파운데이션 모델 (기말 데모에서 실제 사용)


| #   | 모델                  | HF ID / 출처                            | 역할                                     |
| --- | ------------------- | ------------------------------------- | -------------------------------------- |
| 1   | SAM2                | `facebook/sam2-hiera-base-plus`       | Clean Up — 브러시 기반 객체 세그멘테이션            |
| 2   | Qwen2-VL-2B         | `Qwen/Qwen2-VL-2B-Instruct`           | Clean Up — 제거 후 배경 **인페인팅 프롬프트** (VLM) |
| 3   | DreamShaper Inpaint | `Lykon/dreamshaper-8-inpainting`      | Clean Up · Expand · Reframe [완료]       |
| 4   | LaMa                | `big-lama` (`simple-lama-inpainting`) | Expand — 실시간 미리보기 배경                   |
| 5   | Apple SHARP         | `apple/ml-sharp` + 공개 체크포인트           | Reframe — 단일 사진 → 3D Gaussian          |
| 6   | gsplat              | CUDA wheel (torch 2.4.1)              | Reframe — Gaussian splat 렌더            |


**모델 결합:** Clean Up에서 VLM이 장면을 묘사해 DreamShaper 프롬프트를 생성 → 단순 파이프라인 나열이 아니라 **VLM이 인페인팅 품질에 기여**. Expand는 LaMa(빠른 미리보기) + DreamShaper(고품질 확정). Reframe은 SHARP+gsplat(시점) + DreamShaper(빈 영역).

### 성능 팁

- **Reframe:** yaw×pitch 격자 **전체 프리렌더** → CUDA에서 수 분 소요. 상수: `features/reframe.py`, `reframe_yaw.py`.
- **VRAM:** 기능 전환 시 이전 모델 unload. [완료] 시 DreamShaper 로드 + 수십 초.
- **Windows `.venv`:** Python 3.10 · torch **2.4.1+cu124** · diffusers **0.36.0** (`scripts\setup.ps1`).

---

## Run — 중간 과제 단독 테스트

저장소의 `sample.jpg`로 각 foundation model을 개별 실행할 수 있습니다.

```cmd
python common.py
python task_detection_groundingdino.py --image sample.jpg --prompt "person . laptop . bottle ."
python task_segmentation_sam2.py --image sample.jpg --mode auto
python task_vlm_qwen2vl.py --image sample.jpg --question "Describe this image."
python task_depth_depthanythingv2.py --image sample.jpg
python task_pose_vitpose.py --image sample.jpg --score-threshold 0.6
```

결과는 `outputs/` 폴더에 저장됩니다(`.gitignore` 제외 — 로컬 실행 시 생성). `python <script> --help`로 옵션 확인.

---

## Brief explanation of each code

중간 과제 요구사항(**각 코드에 대한 간단한 설명: 입력, 출력, 예시 결과**) + 기말 제출용 `**demo.py`** 설명입니다.

### `demo.py` — 기말 메인 데모 (OpenEdit)


|        |                                                                              |
| ------ | ---------------------------------------------------------------------------- |
| **역할** | Gradio 웹 UI로 **Clean Up · Expand · Reframe** 통합 실행. `ui.py` + `features/` 호출 |
| **입력** | 브라우저에서 **사진 업로드** · 브러시/슬라이더 조작 · [완료] 클릭                                    |
| **출력** | 실시간 미리보기 · [완료] 후 편집 결과(Gradio 다운로드)                                         |
| **예시** | `python demo.py` → `http://127.0.0.1:7860` · `sample.jpg` 업로드 후 기능 시연        |


### `common.py`


|        |                                                                                                |
| ------ | ---------------------------------------------------------------------------------------------- |
| **역할** | 모든 `task_*.py` 공유 유틸: device(`cuda`/`mps`/`cpu`), 이미지 로드·resize, `outputs/` 저장, MPS float64 방지 |
| **입력** | 직접 과제 입력 없음. `load_image(path)` 등으로 간접 호출                                                      |
| **출력** | `device_info()` 콘솔 요약. 그림 저장은 각 `task_*.py`가 처리                                                |
| **예시** | `python common.py` → device·dtype·출력 폴더 경로                                                     |


### `task_detection_groundingdino.py` — Grounding DINO


|        |                                                                                                                           |
| ------ | ------------------------------------------------------------------------------------------------------------------------- |
| **입력** | `--image` 필수. `--prompt`: `"person . laptop ."` 형식. `--box-threshold`, `--text-threshold`, `--no-show`                    |
| **출력** | 콘솔: 클래스·점수·박스. 파일: 탐지 결과 PNG                                                                                              |
| **예시** | `python task_detection_groundingdino.py --image sample.jpg --prompt "person . laptop ."` → `outputs/sample_detection.png` |


### `task_segmentation_sam2.py` — SAM 2


|        |                                                                                                 |
| ------ | ----------------------------------------------------------------------------------------------- |
| **입력** | `--image` 필수. `--mode`: `auto` 또는 `point`. `--no-show`                                          |
| **출력** | 콘솔: 마스크 개수·면적·score. 파일: 마스크 시각화 PNG                                                            |
| **예시** | `--mode auto` → `outputs/sample_seg_auto.png` · `--mode point` → `outputs/sample_seg_point.png` |


### `task_vlm_qwen2vl.py` — Qwen2-VL-2B


|        |                                                                                            |
| ------ | ------------------------------------------------------------------------------------------ |
| **입력** | `--image` 필수. `--question`, `--qa`, `--max-tokens`                                         |
| **출력** | 콘솔: Q&A. 파일: `.txt` + 이미지·Q&A 패널 `.png`                                                    |
| **예시** | `--question "What is in this image?"` → `outputs/sample_vlm.txt`, `outputs/sample_vlm.png` |


### `task_depth_depthanythingv2.py` — Depth Anything V2


|        |                                                                                        |
| ------ | -------------------------------------------------------------------------------------- |
| **입력** | `--image` 필수. `--colormap`(기본 `plasma`), `--no-show`                                   |
| **출력** | 콘솔: 깊이 통계. 파일: 원본 + 깊이 맵 PNG                                                           |
| **예시** | `python task_depth_depthanythingv2.py --image sample.jpg` → `outputs/sample_depth.png` |


### `task_pose_vitpose.py` — ViTPose


|        |                                                     |
| ------ | --------------------------------------------------- |
| **입력** | `--image` 필수. `--score-threshold`, `--no-show`      |
| **출력** | 콘솔: 키포인트 수. 파일: 스켈레톤 오버레이 PNG                       |
| **예시** | `--score-threshold 0.6` → `outputs/sample_pose.png` |


### 요약 표 (`sample.jpg` 기준)


| Script                            | Example command                     | Output under `outputs/`            |
| --------------------------------- | ----------------------------------- | ---------------------------------- |
| `task_detection_groundingdino.py` | `--prompt "person ."`               | `sample_detection.png`             |
| `task_segmentation_sam2.py`       | `--mode auto`                       | `sample_seg_auto.png`              |
| `task_segmentation_sam2.py`       | `--mode point`                      | `sample_seg_point.png`             |
| `task_vlm_qwen2vl.py`             | `--question "Describe this image."` | `sample_vlm.txt`, `sample_vlm.png` |
| `task_depth_depthanythingv2.py`   | (없음)                                | `sample_depth.png`                 |
| `task_pose_vitpose.py`            | (없음)                                | `sample_pose.png`                  |


**Three foundation models:** 서로 다른 foundation model을 쓰는 `task_*.py`가 **5개**이므로 중간 과제 최소 3개 요건 충족.

---

## 파일 구조

```
deeplearning/
  demo.py                # 기말 메인 데모 (제출·발표 진입점)
  app.py                 # demo.py 호환 별칭
  ui.py                  # Gradio UI
  features/
    shared.py            # VLM 캡션 · 인페인팅 합성
    clean_up.py          # Clean Up (SAM2 + Qwen2-VL + DreamShaper)
    expand.py            # Expand (LaMa + DreamShaper)
    reframe.py           # Reframe (SHARP + gsplat + DreamShaper)
  reframe_yaw.py         # yaw×pitch 프리렌더 그리드
  sharp_render.py        # SHARP + gsplat 렌더
  splat_torch.py         # gsplat 불가 시 PyTorch 폴백
  inpaint.py             # 인페인팅 팩토리 (sd15 / lama)
  task_*.py              # 중간 과제 단독 스크립트
  task_inpaint_sd15.py   # DreamShaper SD1.5 inpaint
  task_inpaint_lama.py   # LaMa inpaint
  scripts/setup.ps1      # Windows: ml-sharp clone + Python 3.10 + .venv
  scripts/setup.sh       # Linux/macOS: ml-sharp clone + venv + pip
  requirements.txt
  third_party/ml-sharp   # git clone (setup.ps1 / setup.sh — 레포 미포함)
  sample.jpg / sample2.jpg / sample3.jpg / sample5.jpg
  outputs/               # 결과 (git 제외)
```

---

## 사용 모델 (전체)


| 태스크             | 모델                      | Hugging Face ID                             | 크기(대략) |
| --------------- | ----------------------- | ------------------------------------------- | ------ |
| Detection (중간)  | Grounding DINO base     | `IDEA-Research/grounding-dino-base`         | ~811MB |
| Segmentation    | SAM 2 Hiera base+       | `facebook/sam2-hiera-base-plus`             | ~320MB |
| VLM             | Qwen2-VL-2B-Instruct    | `Qwen/Qwen2-VL-2B-Instruct`                 | ~4GB   |
| Depth (중간)      | Depth Anything V2 Small | `depth-anything/Depth-Anything-V2-Small-hf` | ~97MB  |
| Pose (중간)       | ViTPose base            | `usyd-community/vitpose-base-simple`        | ~330MB |
| Inpaint         | DreamShaper 8 Inpaint   | `Lykon/dreamshaper-8-inpainting`            | ~2GB   |
| Inpaint preview | LaMa                    | `big-lama`                                  | ~200MB |
| NVS (Reframe)   | Apple SHARP             | `apple/ml-sharp`                            | ~수백MB  |


모델은 첫 실행 시 Hugging Face Hub에서 **토큰 없이** 자동 다운로드됩니다.