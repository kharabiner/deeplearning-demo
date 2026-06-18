"""
features — OpenEdit 기능별 코드 패키지

iOS 27 사진편집 3기능을 기능별 모듈로 분리:
  - clean_up : 브러시/텍스트로 선택한 객체 제거 (SAM2 + SD2)
  - expand   : 프레임 축소 → 바깥 영역 아웃페인팅 (LDI + SD2)
  - reframe  : 시점 변경 (SHARP 3DGS + LDI 미리보기 + SD2)

shared.py 는 세 기능이 공유하는 글루 코드(VLM 캡션, 깊이, 인페인팅 합성 등).
"""
