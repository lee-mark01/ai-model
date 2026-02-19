"""
3클래스 해충 탐지 FastAPI 서버
- cockroach (0)
- ant (1)
- spider (2)

사용법:
1. 서버 실행: uvicorn main:app --reload
2. API 문서: http://localhost:8000/docs
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from pathlib import Path

app = FastAPI(
    title="Pest Detection API",
    description="3클래스 해충 탐지 API (cockroach, ant, spider)",
    version="1.0.0"
)

# CORS 설정 (프론트엔드 연동 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드 (서버 시작 시 1회만)
MODEL_PATH = "models/best.pt"
print(f"✅ 모델 로드: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# 클래스 이름 매핑
CLASS_NAMES = {
    0: "cockroach",
    1: "ant",
    2: "spider"
}

# 클래스별 색상
CLASS_COLORS = {
    0: "red",      # cockroach
    1: "green",    # ant
    2: "blue"      # spider
}

@app.get("/")
async def root():
    """API 루트"""
    return {
        "message": "Pest Detection API",
        "model": MODEL_PATH,
        "classes": CLASS_NAMES,
        "endpoints": {
            "predict": "/predict (POST)",
            "predict_json": "/predict/json (POST)",
            "health": "/health (GET)"
        }
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    이미지 업로드 → 해충 탐지 → 바운딩 박스 그린 이미지 + JSON 반환

    Returns:
        - detections: 탐지된 해충 정보 리스트
        - image_base64: 바운딩 박스가 그려진 이미지 (base64)
        - count: 탐지된 해충 개수
    """
    try:
        # 이미지 로드
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 모델 추론
        results = model(image, conf=0.4)  # 신뢰도 임계값 0.4

        # 바운딩박스 그리기 및 JSON 정보 추출
        predictions = []
        draw = ImageDraw.Draw(image)

        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            label_idx = int(box.cls[0])
            score = float(box.conf[0])

            # 클래스 이름
            class_name = CLASS_NAMES.get(label_idx, f"unknown_{label_idx}")
            color = CLASS_COLORS.get(label_idx, "yellow")

            predictions.append({
                "box": {
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3]
                },
                "class_id": label_idx,
                "class_name": class_name,
                "confidence": round(score, 3)
            })

            # 박스 그리기
            draw.rectangle(xyxy, outline=color, width=3)

            # 라벨 텍스트
            label_text = f"{class_name} {score:.2f}"
            draw.text((xyxy[0], xyxy[1] - 10), label_text, fill=color)

        # 결과 이미지 base64로 인코딩
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # JSON 응답 생성
        return JSONResponse(content={
            "success": True,
            "detections": predictions,
            "count": len(predictions),
            "image_base64": img_base64
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@app.post("/predict/json")
async def predict_json_only(file: UploadFile = File(...)):
    """
    이미지 업로드 → 해충 탐지 → JSON만 반환 (이미지 제외, 속도 빠름)

    Returns:
        - detections: 탐지된 해충 정보 리스트
        - count: 탐지된 해충 개수
    """
    try:
        # 이미지 로드
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 모델 추론
        results = model(image, conf=0.4)

        # JSON 정보 추출
        predictions = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            label_idx = int(box.cls[0])
            score = float(box.conf[0])
            class_name = CLASS_NAMES.get(label_idx, f"unknown_{label_idx}")

            predictions.append({
                "box": {
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3]
                },
                "class_id": label_idx,
                "class_name": class_name,
                "confidence": round(score, 3)
            })

        return JSONResponse(content={
            "success": True,
            "detections": predictions,
            "count": len(predictions)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("Pest Detection API 서버 시작")
    print("="*70)
    print(f"모델: {MODEL_PATH}")
    print(f"API 문서: http://localhost:8000/docs")
    print(f"ReDoc: http://localhost:8000/redoc")
    print("="*70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
