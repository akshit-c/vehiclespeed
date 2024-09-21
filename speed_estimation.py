import numpy as np
from collections import deque
import cv2
from scipy.optimize import linear_sum_assignment

# Constants
FRAME_RATE = 30  # Frames per second
FRAME_TIME = 1 / FRAME_RATE
HISTORY_SIZE = 30  # Number of recent positions to keep for each vehicle
MAX_SPEED = 200  # Maximum speed in km/h
MIN_DETECTION_FRAMES = 5  # Minimum number of frames a vehicle must be detected to estimate speed
SPEED_SMOOTHING_FACTOR = 0.3  # Factor for exponential moving average
MAX_TRACKING_DISTANCE = 100  # Maximum distance (in pixels) to associate detections between frames

# These values need to be calibrated for your specific video
DISTANCE_REAL_WORLD = 20  # meters, the real-world distance covered in your region of interest
PIXELS_PER_METER = None  # This will be calculated based on the calibration

class VehicleTracker:
    def __init__(self, vehicle_id, initial_position, initial_frame):
        self.vehicle_id = vehicle_id
        self.positions = deque(maxlen=HISTORY_SIZE)
        self.positions.append((initial_position, initial_frame))
        self.speed = 0
        self.last_position = initial_position
        self.last_frame = initial_frame
        self.lost_frames = 0

    def update_position(self, new_position, frame_number):
        self.positions.append((new_position, frame_number))
        self.last_position = new_position
        self.last_frame = frame_number
        self.lost_frames = 0

    def estimate_speed(self):
        if len(self.positions) < MIN_DETECTION_FRAMES:
            return 0

        distances = []
        time_diffs = []
        for i in range(1, len(self.positions)):
            start_pos, start_frame = self.positions[i-1]
            end_pos, end_frame = self.positions[i]
            distance_pixels = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
            distances.append(distance_pixels / PIXELS_PER_METER)
            time_diffs.append((end_frame - start_frame) * FRAME_TIME)

        if sum(time_diffs) > 0:
            avg_speed = sum(distances) / sum(time_diffs) * 3.6  # Convert to km/h
        else:
            avg_speed = 0

        self.speed = SPEED_SMOOTHING_FACTOR * avg_speed + (1 - SPEED_SMOOTHING_FACTOR) * self.speed
        return min(round(self.speed, 1), MAX_SPEED)

def calibrate_speed_estimation(frame, roi_points):
    global PIXELS_PER_METER
    roi_pixel_length = np.linalg.norm(np.array(roi_points[1]) - np.array(roi_points[0]))
    PIXELS_PER_METER = roi_pixel_length / DISTANCE_REAL_WORLD
    print(f"Calibrated: {PIXELS_PER_METER} pixels per meter")

def associate_detections_to_trackers(detections, trackers):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    cost_matrix = np.zeros((len(detections), len(trackers)))
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            cost_matrix[d, t] = np.linalg.norm(det - trk.last_position)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_indices = np.column_stack((row_ind, col_ind))
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    
    matches = []
    for m in matched_indices:
        if cost_matrix[m[0], m[1]] > MAX_TRACKING_DISTANCE:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def estimate_speed(frame, vehicles, frame_number, roi_points):
    global PIXELS_PER_METER
    
    if PIXELS_PER_METER is None:
        calibrate_speed_estimation(frame, roi_points)

    detections = np.array([[((v[0] + v[2]) / 2, (v[1] + v[3]) / 2)] for v in vehicles])
    
    if frame_number == 0:
        trackers = [VehicleTracker(i, det, frame_number) for i, det in enumerate(detections)]
    else:
        # Initialize trackers if it doesn't exist
        if 'trackers' not in locals():
            trackers = []

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trackers)
        
        # Update matched trackers
        for m in matched:
            trackers[m[1]].update_position(detections[m[0]], frame_number)
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trackers.append(VehicleTracker(len(trackers), detections[i], frame_number))
        
        # Remove lost trackers
        trackers = [t for i, t in enumerate(trackers) if i not in unmatched_trks]

    speeds = {}
    for i, tracker in enumerate(trackers):
        speed = tracker.estimate_speed()
        speeds[f"vehicle_{i}"] = speed

    if speeds:
        print(f"Frame {frame_number}: Speeds: {speeds}")

    return speeds, trackers

def draw_speed_info(frame, vehicles, speeds):
    for i, vehicle in enumerate(vehicles):
        x1, y1, x2, y2, conf, cls = vehicle
        vehicle_id = f"vehicle_{i}"
        speed = speeds.get(vehicle_id, 0)
        
        speed_text = f"{speed:.1f} km/h"
        text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = int(x2) - text_size[0] - 5
        text_y = int(y1) + text_size[1] + 5
        
        cv2.putText(frame, speed_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return frame