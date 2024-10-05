"""This file train.py use for training model"""

# | import
import sys
from ultralytics import YOLO

# | YOLO Training

yolo_model = sys.argv[1] #In YOLOv8 you have a model size n s m l and have 2 option 1.use pre-trained(.pt) 2.build from scratch(.yaml) ex. yolov8n.yaml
data_path = sys.argv[2]  # Dataset_path for training model --> data. ex."Cat_and_Dog_Classification.v1-catndog_dataset.yolov8-obb/data.yaml"
epochs = int(sys.argv[3]) # Number of epochs that you want to train

model = YOLO(yolo_model)

results = model.train(
    data=data_path,
    epochs=epochs,
    imgsz=640,
    device="mps",
)

success = model.export(format="onnx") # Export the model to .onnx extension 
