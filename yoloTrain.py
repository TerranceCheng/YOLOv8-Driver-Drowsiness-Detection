# !!!!!!!!!!! CPU !!!!!!!!!!!

from ultralytics import YOLO

# Create model
model = YOLO('yolov8n.pt')

# Train model
results = model.train(data='.\data.yaml', epochs=1, save_period=-1)
# model.export(format="onnx",opset=12)
