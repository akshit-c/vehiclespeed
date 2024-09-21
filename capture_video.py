import cv2
from vehicle_detection import detect_vehicles

def process_video_stream():
    cap = cv2.VideoCapture('/Users/amit/Downloads/Traffic IP Camera video.mp4')# For live webcam input, replace with file path for video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Process each frame
        detect_vehicles(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
