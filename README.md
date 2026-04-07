# Deep Learning Assignment — Foundation Model Tasks

Hugging Face foundation model을 활용한 중간/기말 과제 코드입니다.  
**Windows / Linux / macOS(Apple Silicon M2) 모두 지원합니다.**

---

## 빠른 시작 (clone → 실행)

```bash
git clone <repo-url>
cd deeplearning

# 가상환경 생성 (선택, 권장)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# SAM2는 별도 설치 필요
pip install git+https://github.com/facebookresearch/sam2.git

# 환경 확인 (device, dtype 출력)
python common.py

# 테스트 실행 (sample.jpg를 본인 이미지로 교체)
python task_depth.py --image sample.jpg
```

---

## 파일 구조

```
deeplearning/
  common.py              # 공통 유틸: device, resize, float32 안전 변환
  task_detection.py      # Grounding DINO — open-vocab 탐지
  task_segmentation.py   # SAM2 — 픽셀 세그멘테이션
  task_vlm.py            # Qwen2-VL-2B — 이미지 질의응답
  task_depth.py          # Depth Anything V2 — 깊이 추정
  task_pose.py           # ViTPose — 관절 키포인트
  pipeline_final.py      # 기말용: 전체 파이프라인
  requirements.txt
  .gitignore
  outputs/               # 결과 자동 저장 (gitignore 처리됨)
```

---

## 각 태스크 실행 방법

```bash
# 1. 환경 확인
python common.py

# 2. Open-Vocabulary Detection
python task_detection.py --image sample.jpg --prompt "person . laptop . bottle ."

# 3. Segmentation (자동 전체 분할)
python task_segmentation.py --image sample.jpg --mode auto

# 3. Segmentation (이미지 중앙 포인트)
python task_segmentation.py --image sample.jpg --mode point

# 4. VLM 단일 질문
python task_vlm.py --image sample.jpg --question "Describe this image."

# 4. VLM 기본 5개 질문 연속 실행
python task_vlm.py --image sample.jpg --qa

# 5. Depth Estimation
python task_depth.py --image sample.jpg

# 6. Pose Estimation
python task_pose.py --image sample.jpg

# 기말: 전체 파이프라인 (이미지 1장 → 전체 분석 → 보고서)
python pipeline_final.py --image sample.jpg
```

결과 이미지는 `outputs/` 폴더에 자동 저장됩니다.

---

## 하드웨어 지원

| 환경 | Device | dtype | 비고 |
|---|---|---|---|
| NVIDIA RTX 3070 (Windows/Linux) | `cuda` | `float16` | 가장 빠름 |
| Apple M2 Air (macOS) | `mps` | `float32` | MPS는 float64/float16 미지원 |
| CPU (어디서나) | `cpu` | `float32` | 느리지만 모든 환경에서 동작 |

`common.py`에서 자동 감지하므로 **코드 수정 없이** 환경에 따라 자동 전환됩니다.

```python
# common.py 핵심 — MPS float64 방지
torch.set_default_dtype(torch.float32)  # 전역 float32 강제

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"
```

---

## 이미지 입력 주의사항

- 모든 이미지는 `load_image()` 함수를 통해 로드됩니다
- **가장 긴 변이 1333px 초과 시 자동 축소** (비율 유지)
- 너무 큰 이미지는 OOM 또는 처리 시간 초과의 원인이 됩니다
- 지원 형식: JPG, PNG, WEBP, BMP 등 PIL이 지원하는 모든 형식

---

## 사용 모델

| 태스크 | 모델 | HuggingFace ID | 크기 |
|---|---|---|---|
| Detection | Grounding DINO base | `IDEA-Research/grounding-dino-base` | ~811MB |
| Segmentation | SAM2 hiera base+ | `facebook/sam2-hiera-base-plus` | ~320MB |
| VLM | Qwen2-VL-2B Instruct | `Qwen/Qwen2-VL-2B-Instruct` | ~4GB |
| Depth | Depth Anything V2 Small | `depth-anything/Depth-Anything-V2-Small-hf` | ~97MB |
| Pose | ViTPose base | `usyd-community/vitpose-base-simple` | ~330MB |

> 모델은 첫 실행 시 Hugging Face에서 자동 다운로드됩니다.  
> 다운로드 위치: `~/.cache/huggingface/` (기본값)

---

## 트러블슈팅

### MPS 관련 에러 (Apple Silicon)
```
RuntimeError: MPS does not support float64
```
→ `common.py`의 `torch.set_default_dtype(torch.float32)` 가 적용되어 있는지 확인  
→ numpy 배열 생성 시 반드시 `dtype=np.float32` 명시

### CUDA OOM (RTX 3070)
```
torch.cuda.OutOfMemoryError
```
→ `pipeline_final.py`는 모델 간 `del model` + `torch.cuda.empty_cache()` 처리가 되어 있음  
→ VLM만 단독 실행 시 OOM이면 `--max-tokens` 줄이거나 Qwen2-VL-2B → moondream2로 교체

### SAM2 설치 오류
```
ModuleNotFoundError: No module named 'sam2'
```
→ `pip install git+https://github.com/facebookresearch/sam2.git` 재시도  
→ 안 되면: `pip install sam2` (PyPI 버전)

### Grounding DINO 빌드 오류 (Windows)
```
error: Microsoft Visual C++ 14.0 is required
```
→ Visual Studio Build Tools 설치: https://visualstudio.microsoft.com/downloads/  
→ 또는 Conda 환경에서 설치 권장
