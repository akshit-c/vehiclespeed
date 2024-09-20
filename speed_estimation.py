import cv2
import numpy as np

def estimate_speed(video_path, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video file")
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    speeds = []
    frame_count = 0

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Process every 5th frame to reduce computation
        if frame_count % 5 != 0:
            continue

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate average movement
        movement = np.mean(np.abs(flow))

        # Calculate speed (pixels per second)
        speed = movement * frame_rate

        speeds.append(speed)
        
        prev_gray = current_gray

    cap.release()

    # Return average speed if speeds were calculated
    return np.mean(speeds) if speeds else None

# Example usage
if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"
    avg_speed = estimate_speed(video_path)
    if avg_speed is not None:
        print(f"Average speed: {avg_speed:.2f} pixels per second")
    else:
        print("Failed to estimate speed")