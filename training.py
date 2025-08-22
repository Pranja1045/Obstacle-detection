from ultralytics import YOLO

model=YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=100, imgsz=640)
model.val()
while True:
    _, frame = cap.read()
    name = './data/frame' + str(currentFrame) + '.jpg'
    cv2.imwrite(name, frame)
    cv2.imshow("Original Frame", frame)
    model.predict(source=name, save=True)
    currentFrame += 1
model.export(format='onnx')