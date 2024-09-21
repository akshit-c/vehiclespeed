import cv2
import numpy as np
from sklearn.cluster import KMeans
import logging

def rgb_to_hsv(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]

def hsv_to_name(hsv):
    h, s, v = hsv
    s = s * 100 / 255  
    v = v * 100 / 255  

    if s <= 15 and v >= 85:
        return 'White'
    elif v <= 25:
        return 'Black'
    elif s <= 15 and 25 < v < 85:
        return 'Gray'
    elif h <= 10 or h > 350:
        return 'Red'
    elif 10 < h <= 30:
        return 'Orange'
    elif 30 < h <= 70:
        return 'Yellow'
    elif 70 < h <= 165:
        return 'Green'
    elif 165 < h <= 260:
        return 'Blue'
    elif 260 < h <= 315:
        return 'Purple'
    elif 315 < h <= 350:
        return 'Pink'
    else:
        return 'Unknown'

def detect_color(frame, roi=None):
    logging.info("Detecting color in frame")
    
    if roi:
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]
    
    # Convert frame to HSV (Hue, Saturation, Value)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Reshape the frame to be a list of pixels
    pixels = hsv_frame.reshape((-1, 3))
    
    # Perform K-means clustering to find the most dominant colors
    kmeans = KMeans(n_clusters=5, n_init=10)
    kmeans.fit(pixels)
    
    # Get the HSV values of the dominant colors
    dominant_colors = kmeans.cluster_centers_
    
    # Get the percentage of each color
    labels = kmeans.labels_
    color_percentages = np.bincount(labels) / len(labels)
    
    # Sort colors by percentage in descending order
    sorted_colors = sorted(zip(dominant_colors, color_percentages), key=lambda x: x[1], reverse=True)
    
    # Convert the dominant colors to names and filter out 'Unknown'
    color_names = [hsv_to_name(color) for color, _ in sorted_colors if hsv_to_name(color) != 'Unknown']
    
    # Return the most dominant color
    dominant_color = color_names[0] if color_names else 'Unknown'
    
    logging.info(f"Detected color: {dominant_color}")
    return dominant_color

def detect_colors(frame, roi=None):
    return detect_color(frame, roi)