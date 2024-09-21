import cv2
import numpy as np
from vehicle_detection import detect_vehicles
import time
import ssl
from number_plate_recognition import recognize_number_plate
from speed_estimation import estimate_speed, draw_speed_info, calibrate_speed_estimation
from color_detection import detect_color
import sys

ssl._create_default_https_context = ssl._create_unverified_context

def process_video_realtime(input_path):
    cap = cv2.VideoCapture(input_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_number = 0

    roi_start = int(frame_height * 0.5)
    roi_end = int(frame_height * 0.8)
    roi_points = [(0, roi_start), (frame_width, roi_end)]

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return
    calibrate_speed_estimation(first_frame, roi_points)

    trackers = []

    cv2.namedWindow('Vehicle Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Vehicle Detection', 1280, 720)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        vehicles = detect_vehicles(frame)
        speeds, trackers = estimate_speed(frame, vehicles, frame_number, roi_points)

        cv2.line(frame, (0, roi_start), (frame_width, roi_start), (0, 255, 0), 2)
        cv2.line(frame, (0, roi_end), (frame_width, roi_end), (0, 255, 0), 2)

        for i, vehicle in enumerate(vehicles):
            x1, y1, x2, y2, conf, cls = vehicle

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            vehicle_img = frame[int(y1):int(y2), int(x1):int(x2)]
            color = detect_color(vehicle_img)

            plate_number, plate_roi = recognize_number_plate(frame, vehicle)

            if plate_roi is not None:
                px1, py1, px2, py2 = plate_roi
                cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 2)

            class_name = get_class_name(cls)
            info = f"{class_name}: {conf:.2f}, Color: {color}"
            cv2.putText(frame, info, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(frame, f"Plate: {plate_number}", (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        frame = draw_speed_info(frame, vehicles, speeds)

        cv2.imshow('Vehicle Detection', frame)

        process_time = time.time() - start_time
        wait_time = max(1, int((frame_time - process_time) * 1000))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    print("Video processing complete.")

def get_class_name(class_id):
    class_names = {2: 'Car', 3: 'Motorbike', 5: 'Bus', 7: 'Truck'}
    return class_names.get(class_id, 'Unknown')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_video_path = sys.argv[1]
        process_video_realtime(input_video_path)
    else:
        print("No input video path provided")