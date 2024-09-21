import cv2
import logging

def capture_frames(video_path):
    logging.info(f"Attempting to capture frames from {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        logging.error(f"Error: Unable to open video file {video_path}")
        return frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    logging.info(f"Captured {len(frames)} frames from {video_path}")
    return frames
