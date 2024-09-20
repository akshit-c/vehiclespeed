import numpy as np
import cv2

cap = cv2.VideoCapture('/Users/amit/Downloads/2034115-hd_1920_1080_30fps.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the bounding box coordinates from the previous step
    (x, y, w, h) = cv2.boundingRect(frame)

    # Extract the vehicle region
    roi = frame[y:y+h, x:x+w]

    # Convert the ROI to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Calculate the dominant color in the HSV space
    dominant_color = np.argmax(np.bincount(hsv[:, :, 0].flatten()))

    # Map the dominant color to a color name
    color_names = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Purple']
    color_name = color_names[dominant_color]

    # Print the detected color
    print(f"Detected color: {color_name}")

    # Display the output
    cv2.imshow('Color Detection', roi)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break