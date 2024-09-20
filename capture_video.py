import cv2

def capture_video():
    # Your video capture code here
    pass

# If it's not a function but a variable, it should be defined like this:
# capture_video = ...

# Capture video from camera (0) or load a video file
cap = cv2.VideoCapture('/Users/amit/Downloads/2034115-hd_1920_1080_30fps.mp4')  # Replace 0 with the video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the grayscale frame
    cv2.imshow('Video', gray)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
