import cv2
import numpy as np
import yolov5

# Load the YOLOv5 model
model = yolov5.load('yolov5s.pt')
net = cv2.dnn.readNetFromDarknet('yolov5s.cfg', 'yolov5s.weights')

# Initialize video capture
cap = cv2.VideoCapture('/Users/amit/Downloads/2034115-hd_1920_1080_30fps.mp4')  # Use 0 for webcam or provide a video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the frame dimensions
    (H, W) = frame.shape[:2]

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # Set the input for the YOLOv5 model
    net.setInput(blob)

    # Run the YOLOv5 model
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Loop through the detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Filter for vehicles (class_id 0)
                # Get the bounding box coordinates
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Draw the bounding box
                cv2.rectangle(frame, (centerX, centerY), (centerX + width, centerY + height), (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Vehicle Detection', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object when done
cap.release()
cv2.destroyAllWindows()