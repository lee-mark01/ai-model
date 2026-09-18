from ultralytics import YOLO
from pathlib import Path

# 1. 모델 로드 (best.pt 우선, 없으면 stable 사용)
model_path = 'models/best.pt'
if not Path(model_path).exists():
    model_path = 'pest_detection/yolov8x_stable/weights/best.pt'
best_model = YOLO(model_path)

# 2. 폴더 전체 테스트 및 결과 저장 (TTA 적용)
results = best_model.predict(
    source='dataset/test/images',
    conf=0.35,       # TTA 사용 시 base conf 낮춤
    augment=True,    # TTA: 이미지를 여러 방향으로 augment 후 예측 통합
    save=True,
    project='results',
    name='test_output_tta'
)
