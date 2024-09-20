import cv2
import numpy as np

def detect_color(vehicle):
    # Convert to HSV color space
    hsv = cv2.cvtColor(vehicle, cv2.COLOR_BGR2HSV)

    # Calculate histogram
    hist, _ = np.histogram(hsv[:, :, 0], 256, [0, 256])

    # Find dominant color
    dominant_color = np.argmax(hist)

    # Convert to color name
    color_names = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
    color = color_names[dominant_color // 30]

    return color