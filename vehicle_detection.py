from ultralytics import YOLO
import cv2
import numpy as np

# Load the model only once
model = YOLO('yolov8x.pt')  # Using YOLOv8x for better accuracy

def detect_vehicles(frame):
    # Perform inference
    results = model(frame, conf=0.5)  # Increased confidence threshold

    vehicles = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf.item()
            cls = int(box.cls.item())
            if cls in [2, 3, 5, 7]:  # car, motorbike, bus, truck
                # Apply non-maximum suppression
                if not is_overlapping(vehicles, (x1, y1, x2, y2)):
                    vehicles.append((x1, y1, x2, y2, conf, cls))

    return vehicles

def is_overlapping(existing_vehicles, new_vehicle):
    for vehicle in existing_vehicles:
        iou = calculate_iou(vehicle[:4], new_vehicle)
        if iou > 0.5:  # If IoU is greater than 0.5, consider it as overlapping
            return True
    return False

def calculate_iou(box1, box2):
    # Calculate intersection over union
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0