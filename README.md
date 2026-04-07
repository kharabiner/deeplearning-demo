# Deep Learning Assignment — Foundation Model Tasks

Hugging Face foundation model 기반 중간/기말 과제 코드입니다.

---

## 설치 및 실행

```cmd
git clone <repo-url>
cd deeplearning

python -m venv .venv
.venv\Scripts\activate.bat

.venv\Scripts\python.exe -m pip install -r requirements.txt

python common.py
python task_detection.py --image sample.jpg --prompt "person . laptop . bottle ."
python task_segmentation.py --image sample.jpg --mode auto
python task_vlm.py --image sample.jpg --question "Describe this image."
python task_depth.py --image sample.jpg
python task_pose.py --image sample.jpg
```

결과는 `outputs/` 폴더에 자동 저장됩니다.

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
  outputs/               # 결과 저장 폴더 (git 제외)
```

---

## 사용 모델

| 태스크 | HuggingFace ID | 크기 |
|---|---|---|
| Detection | `IDEA-Research/grounding-dino-base` | ~811MB |
| Segmentation | `facebook/sam2-hiera-base-plus` | ~320MB |
| VLM | `Qwen/Qwen2-VL-2B-Instruct` | ~4GB |
| Depth | `depth-anything/Depth-Anything-V2-Small-hf` | ~97MB |
| Pose | `usyd-community/vitpose-base-simple` | ~330MB |

모델은 첫 실행 시 자동 다운로드됩니다.
