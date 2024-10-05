"""This predict.py use for predict cat and dog image and save result of prediction to test_predict_res folder"""
#| Import
import sys
from ultralytics import YOLO

#| Predict image
model_path = sys.argv[1] #Firstly type a model path that have a .onnx extension
image_path = sys.argv[2] #Secondly type a string image path
print("Imageee!!!",image_path)
model = YOLO(model_path)

results = model(image_path)  #such as "Pred_Dataset/images/val/*.png"
print(str(image_path))
boxes_cnt = 0
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    boxes_cnt += len(boxes)
    result.save(filename=f"test_predict_res/result_{i:02}.jpg")  # save predict image to folder test_predict_res

print(len(results), boxes_cnt)