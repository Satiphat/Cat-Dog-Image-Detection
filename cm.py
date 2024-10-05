"""Run This File cm.py for compute confusion metric at the each confusion --> return """

import json
import os
import shutil
import sys

import numpy as np

from ultralytics import YOLO

def rm_val(path: str = "runs/detect/val"):
    """remove val dir from runs -> detect"""
    shutil.rmtree(path, ignore_errors=False)


if __name__ == "__main__":
    model_path = sys.argv[1] #First Type model file path
    conf_range = int(sys.argv[2]) #Second Type the confidential range that you want to use
    data_path = sys.argv[3] #Third Type the dataset path --> data.yam

    model = YOLO(model_path, task="detect")
    conf = np.linspace(0.1, 0.99, conf_range)

    res = {}
    try:
        for i, c in enumerate(conf):
            print(f"\nRound #{i+1:03}\nConf.: {c:.3f}\n")

            metrics = model.val(
                data=data_path,
                imgsz=640,
                conf=c,
                # split="val",
                device="mps",
            )
            cm = metrics.confusion_matrix.matrix.tolist()
            res[c] = cm
            rm_val()
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detect! Removing Val director...")
        rm_val()

    print(res)

    model_name = os.path.splitext(os.path.split(model_path)[1])[0]

    with open(f"{model_name}.json", "wt", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
