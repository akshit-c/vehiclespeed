import cv2
import pytesseract
import numpy as np

# Import the functions from each step
from capture_video import capture_video
from detect_vehicle import detect_vehicle
from extract_number_plate import extract_number_plate
from estimate_speed import estimate_speed
from detect_color import detect_color  # Optional

def main():
    # Initialize the video capture
    cap = cv2.VideoCapture('/Users/amit/Downloads/2034115-hd_1920_1080_30fps.mp4')

    while True:
        # Capture and process video input
        frame = capture_video(cap)

        # Apply vehicle detection model
        detections = detect_vehicle(frame)

        # Extract number plate and recognize characters
        number_plate = extract_number_plate(detections)

        # Estimate vehicle speed
        speed = estimate_speed(frame)

        # Detect vehicle color (optional)
        color = detect_color(frame) if detect_color else None

        # Display the results
        cv2.imshow('Video', frame)
        print(f"Number Plate: {number_plate}, Speed: {speed:.2f} km/h, Color: {color}")

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()