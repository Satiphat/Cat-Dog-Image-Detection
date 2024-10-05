"""This val.py use for valid model and return to folder runs"""

#| Import
from ultralytics import YOLO
import sys

# | Val
model_path = sys.argv[1] #Typing the model path that have .onnx in to your terminal like  python val.py model_path
dataset_path = sys.argv[2] #datasets_path must be data.yaml
iou = float(sys.argv[3])
model = YOLO(model=model_path)
result = model.val(data=dataset_path,imgsz=640,device="mps",iou=iou)


