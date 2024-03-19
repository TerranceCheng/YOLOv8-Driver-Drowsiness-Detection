# !!!!!!!!!!! CPU !!!!!!!!!!!

from ultralytics import YOLO

# Create model
model = YOLO('yolov8s.pt')

# Train model
results = model.train(data='.\data.yaml', epochs=2, save_period=-1)
# model.export(format="onnx",opset=12)
