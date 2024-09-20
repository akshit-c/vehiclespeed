import numpy as np
import cv2

# Load the pre-recorded video
cap = cv2.VideoCapture('/Users/amit/Downloads/2034115-hd_1920_1080_30fps.mp4')

# Get the video's frame rate
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize the previous frame and timestamp
prev_frame = None
prev_timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the vehicle speed
    speed = np.mean(flow) * fps

    # Print the estimated speed
    print(f"Estimated speed: {speed:.2f} km/h")

    # Update the previous frame and timestamp
    prev_frame = gray
    prev_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

    # Display the output
    cv2.imshow('Speed Estimation', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break