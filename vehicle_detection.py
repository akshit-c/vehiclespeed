import cv2
import numpy as np

def detect_vehicles(frame):
    # Load YOLOv4 model
    net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

    # Get layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Run object detection
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Get detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Get vehicle bounding boxes
    vehicles = []
    for i in indices:
        i = i[0] if isinstance(i, list) else i  # Compatibility with different OpenCV versions
        x, y, w, h = boxes[i]
        vehicles.append(frame[y:y+h, x:x+w])

    return vehicles