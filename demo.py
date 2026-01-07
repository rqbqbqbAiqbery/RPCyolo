from ultralytics import YOLO

model = YOLO("runs/detect/rpc_det/weights/best.pt")
# source=0 表示默认摄像头
model.predict(source=0, conf=0.25, iou=0.7, show=True)