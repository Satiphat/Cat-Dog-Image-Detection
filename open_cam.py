"""This file open_cam.py is use for opening the the webcam that you want for cat and dog detection with the bounding box"""

# | Import 
import cv2
import sys
from ultralytics import YOLO

# | Define name Class and Color for each class
class_labels = {
    0: "cat",
    1: "dog"
}

# Define colors: Red for cat, Blue for dog
colors = {
    0: (0, 0, 255),  # Red in BGR format (for OpenCV)
    1: (255, 0, 0)   # Blue in BGR format (for OpenCV)
}

# | Funtion
def detect_with_webcam(model):
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)  # '0' is the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use YOLOv8 model for detection
        results = model(frame)

        # Loop through detected results and draw bounding boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # xyxy format
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID

                color = colors[cls] #Red for detect cat and Blue for dog

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4) #The border thickness set to 4
                label = f"{class_labels[cls]}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3) #(0,0,0) is a black color

        # Display the frame with detections
        cv2.imshow("YOLOv8 Cat&Dog Webcam Detection", frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# | Main
if __name__ == "__main__":
    model_path = sys.argv[1]
    model = YOLO(model=model_path)
    detect_with_webcam(model)
