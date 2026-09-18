# API 연동 가이드

`models/best.pt` (YOLOv8x, 3-class)를 FastAPI로 서빙하는 방법과 클라이언트 연동 예시입니다.
모델 성능·학습 과정은 [README.md](README.md) 를 참고하세요.

## 1. 서버 실행

```bash
# 개발 모드 (코드 변경 시 자동 재시작)
uvicorn main:app --reload

# 프로덕션 모드
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

모델은 서버 시작 시 1회만 로드됩니다. 요청마다 로드하지 않습니다.

> `--workers N` 은 프로세스를 N개 띄우므로 모델도 N번 메모리에 올라갑니다. YOLOv8x는 가중치만 137MB이고 PyTorch 런타임까지 더하면 프로세스당 메모리 사용량이 상당하니, 서버 메모리를 확인하고 worker 수를 정하세요.

## 2. 엔드포인트

### `GET /`
모델 정보와 클래스 목록을 반환합니다.

```json
{
  "message": "Pest Detection API",
  "model": "models/best.pt",
  "classes": { "0": "cockroach", "1": "ant", "2": "spider" },
  "endpoints": {
    "predict": "/predict (POST)",
    "predict_json": "/predict/json (POST)",
    "health": "/health (GET)"
  }
}
```

### `GET /health`
헬스 체크. 로드밸런서·컨테이너 오케스트레이터의 probe로 사용합니다.

```json
{ "status": "healthy", "model_loaded": true, "model_path": "models/best.pt" }
```

### `POST /predict`
이미지를 업로드하면 탐지 결과와 **바운딩 박스가 그려진 이미지**를 함께 반환합니다.

- Content-Type: `multipart/form-data`
- Field: `file` (이미지 파일)

```json
{
  "success": true,
  "count": 2,
  "detections": [
    {
      "box": { "x1": 123.45, "y1": 67.89, "x2": 234.56, "y2": 178.90 },
      "class_id": 0,
      "class_name": "cockroach",
      "confidence": 0.854
    },
    {
      "box": { "x1": 345.67, "y1": 123.45, "x2": 456.78, "y2": 234.56 },
      "class_id": 1,
      "class_name": "ant",
      "confidence": 0.923
    }
  ],
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### `POST /predict/json`
`/predict` 와 동일하지만 `image_base64` 를 생략합니다. 이미지 인코딩 비용이 없어 응답이 빠르므로, 클라이언트가 직접 박스를 그리는 경우 이쪽을 사용하세요.

### 좌표계

`box` 는 **업로드한 원본 이미지 기준 픽셀 좌표**(`x1, y1` = 좌상단, `x2, y2` = 우하단)입니다. 별도 스케일링 없이 원본 위에 그대로 그리면 됩니다.

### 에러 응답

실패 시 HTTP 500과 함께 다음을 반환합니다.

```json
{ "success": false, "error": "cannot identify image file" }
```

## 3. 클래스별 신뢰도 임계값

클래스마다 학습 데이터 양이 달라 단일 임계값 대신 클래스별 임계값을 적용합니다 (`main.py`).

| class_id | 클래스 | threshold | 의도 |
|---|---|---:|---|
| 0 | cockroach | 0.45 | 데이터 충분 → 오탐 억제 우선 |
| 1 | ant | 0.35 | 균형 |
| 2 | spider | 0.30 | 데이터 부족 → 미탐 방지 우선 |

임계값을 조정하려면 `main.py` 의 `CLASS_CONF_THRESHOLDS` 를 수정하세요. 값을 낮추면 탐지 수가 늘고 오탐도 함께 늘어납니다.

## 4. 클라이언트 예시

### Python

```python
import requests

with open("image.jpg", "rb") as f:
    res = requests.post("http://localhost:8000/predict/json", files={"file": f})

data = res.json()
print(f"탐지 개수: {data['count']}")
for det in data["detections"]:
    print(f"{det['class_name']}: {det['confidence']}")
```

### JavaScript

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const res = await fetch("http://localhost:8000/predict/json", {
  method: "POST",
  body: formData,
});
const data = await res.json();

data.detections.forEach((det) => {
  console.log(`${det.class_name}: ${det.confidence}`);
});
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict/json" -F "file=@image.jpg"
```

## 5. Docker 배포

저장소 루트의 `Dockerfile` 을 그대로 사용합니다. CPU 추론 기준입니다.

```bash
docker build -t pest-detection-api .
docker run -p 8000:8000 pest-detection-api
```

`models/best.pt` 는 Git LFS로 관리되므로, 빌드 전에 LFS 파일이 실제로 받아졌는지 확인하세요. 포인터 파일만 있으면 모델 로드에 실패합니다.

```bash
git lfs pull
ls -lh models/best.pt   # 137MB 이면 정상
```

## 6. 모델 직접 사용

FastAPI 없이 파이썬에서 바로 쓰는 경우입니다.

```python
from ultralytics import YOLO
from PIL import Image

model = YOLO("models/best.pt")          # names: {0: cockroach, 1: ant, 2: spider}
results = model(Image.open("test.jpg"), conf=0.30)

for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"{model.names[int(box.cls[0])]} {float(box.conf[0]):.3f} ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
```

## 7. 대화형 문서

서버 실행 후 접속할 수 있습니다.

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 8. 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| 모델 로드 실패 | LFS 포인터만 받아짐 | `git lfs pull` 후 파일 크기 확인 |
| `ImportError: libGL.so.1` | slim 이미지에 OpenCV 의존성 없음 | `Dockerfile` 의 `libgl1`, `libglib2.0-0` 설치 확인 |
| CUDA OOM | GPU 메모리 부족 | `model(image, device="cpu")` 로 CPU 추론 |
| 응답 지연 | 이미지 base64 인코딩 비용 | `/predict/json` 사용 |

## 9. 프로덕션 적용 시 고려사항

현재 코드는 데모·포트폴리오 수준이며, 실제 서비스에 올린다면 다음이 필요합니다.

- **CORS** — `main.py` 는 `allow_origins=["*"]` 입니다. 배포 도메인으로 제한하세요.
- **업로드 크기 제한** — 현재 제한이 없어 대용량 파일로 메모리를 소진시킬 수 있습니다.
- **인증 / Rate limiting** — 추론 비용이 큰 엔드포인트라 공개 시 남용에 취약합니다.
- **HTTPS**, 구조적 로깅, 에러 추적(Sentry 등), 메트릭 수집

## 10. 라이선스 주의

- **모델 가중치**: Ultralytics YOLOv8 기반이므로 **AGPL-3.0** 이 적용됩니다. 상업적 사용 시 소스 공개 의무가 있거나 Ultralytics Enterprise 라이선스가 필요합니다.
- **데이터셋**: CC BY 4.0 — 재배포 시 출처 표기가 필요합니다.
