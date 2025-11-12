from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import base64

app = FastAPI()
model = YOLO('models/best.pt')  # 모델 경로 적절히 지정

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 이미지 로드
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 모델 추론
    results = model(image)
    
    # 바운딩박스 그리기 및 JSON 정보 추출
    predictions = []
    draw = ImageDraw.Draw(image)
    for box in results[0].boxes:
        xyxy = box.xyxy[0].tolist()
        label_idx = int(box.cls[0])
        score = float(box.conf[0])
        predictions.append({
            "box": xyxy,
            "class": label_idx,
            "score": score
        })
        # 박스 그리기
        draw.rectangle(xyxy, outline="red", width=3)
    
    # 결과 이미지 base64로 인코딩
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # JSON 응답 생성
    return JSONResponse(content={
        "detections": predictions,
        "image_base64": img_base64
    })
