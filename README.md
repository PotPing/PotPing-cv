# 🚗 YOLOv8n 기반 실시간 포트홀 검출 시스템

도로 위의 포트홀(움푹 패인 곳)을 실시간으로 감지하는 딥러닝 프로젝트입니다.

---

## ✨ 주요 기능

- **실시간 포트홀 감지**: 웹캠 또는 비디오 파일에서 실시간으로 포트홀 검출
- **CLAHE 전처리**: 어두운 환경에서도 높은 정확도 유지
- **중복 제거**: 같은 포트홀을 중복 저장하지 않는 스마트 필터링
- **심각도 분류**: LOW/MEDIUM/HIGH 3단계로 위험도 자동 분류
- **백엔드 연동**: 검출 결과를 자동으로 서버에 전송

---

## 📁 프로젝트 구조
PotPing-cv/
├── src/
│   ├── outputs/weights/road_anomaly_yolo_clahe/weights
│   │   ├──best.pt           # 최적 성능 모델
│   │   └──last.pt           # 마지막 epoch 모델
│   ├── results/
│   │   └──live_detections.json # 서버(백엔드)에 전송할 데이터
│   ├── main.py              # 메인 실행 파일
│   └── best_conf.json       # 최적 confidence threshold
├── results/                 # 검출 결과 저장 폴더
├── config.yaml              # 데이터셋 설정
├── requirements.txt         # 필요한 라이브러리
└── README.md                # 이 파일

---

## 🔧 설치 방법

### 1️⃣ 필수 프로그램 설치

#### Python 설치 (3.8 이상)
- [Python 공식 사이트](https://www.python.org/downloads/)에서 다운로드
- 설치 시 **"Add Python to PATH"** 체크 필수!

### 2️⃣ 프로젝트 다운로드

```bash
# 터미널(또는 CMD) 열기
git clone https://github.com/PotPing/PotPing-cv.git
cd PotPing-cv
```

### 3️⃣ 가상환경 생성 및 활성화
**src 경로로 이동 후 수행:**

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4️⃣ 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 5️⃣ 학습된 모델 경로

다운로드 후 다음 경로에 저장:
```
outputs/weights/road_anomaly_yolo_clahe/weights/best.pt
```

---

## 🚀 사용 방법

### 실시간 웹캠 검출 (가장 많이 사용)

```bash
cd src
python main.py --mode live-camera
```

- **종료**: `ESC` 키를 누르세요
- 검출된 포트홀은 `results/` 폴더에 자동 저장됩니다

### 고급 옵션

#### 비디오 파일에서 검출
`main.py` 파일의 `VIDEO_PATH` 변수를 수정:
```python
VIDEO_PATH = "path/to/your/video.mp4"  # 파일 경로 지정
```

### 폴더 이미지 일괄 처리

```bash
python main.py --mode predict
```

---

## 🔌 API 연동

### 백엔드 서버 설정

`main.py` 파일에서 서버 주소 수정:

```python
POST_URL = "http://localhost:8080/api/potholes/detection"  
# ↑ 여기를 실제 서버 주소로 변경
```

### 전송 데이터 형식

```json
{
  "video_timestamp": 1.2,
  "severity": "HIGH",
  "status": "DETECTED",
  "total_detections": 5,
  "average_confidence": 0.87,
  "images": {
    "original": "original_20231203_125533_1.jpg",
    "processed": "enhanced_20231203_125533_1.jpg",
    "detected": "pothole_20231203_125533_1.jpg"
  }
}
```

---

## 🎯 주요 파라미터 설명

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `conf_threshold` | 0.45 | 검출 신뢰도 임계값 (낮을수록 많이 검출) |
| `save_interval_seconds` | 2.0 | 같은 위치 재검출 방지 시간 (초) |
| `min_movement_threshold` | 100.0 | 최소 이동 거리 (픽셀) |

### 설정 변경 방법

`src/main.py` 파일의 `live_detect()` 함수 호출 부분 수정:

```python
live_detect(
    video_path=VIDEO_PATH,
    conf_threshold=0.45,           # 더 많이 검출하려면 낮추기 (예: 0.25)
    save_interval_seconds=2.0,     # 저장 간격 조정
    min_movement_threshold=100.0   # 최소 이동 거리
)
```

---

## ❗ 문제 해결

### Q1. "ModuleNotFoundError: No module named 'ultralytics'"
**해결**: 가상환경이 활성화되지 않았습니다
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Q2. "Model file not found"
**해결**: 학습된 모델을 다운로드하여 올바른 경로에 저장하세요
- 경로: `outputs/weights/road_anomaly_yolo_clahe/weights/best.pt`

### Q3. 웹캠이 실행되지 않음
**해결**: 
1. 다른 프로그램에서 웹캠을 사용 중인지 확인
2. `main.py`에서 `VIDEO_PATH = 0`을 `VIDEO_PATH = 1`로 변경

### Q4. 검출이 너무 많이/적게 됩니다
**해결**: `best_conf.json` 파일의 값을 수정
```json
{
  "best_conf": 0.45  // 더 많이 검출하려면 낮추기 (예: 0.25)
}
```

---

## 📊 검출 결과 확인

검출된 이미지는 `results/` 폴더에 저장됩니다:

- `original_*.jpg`: 원본 이미지
- `enhanced_*.jpg`: CLAHE 전처리 적용
- `pothole_*.jpg`: 검출 박스 표시
- `live_detections.json`: 전체 검출 기록

---
