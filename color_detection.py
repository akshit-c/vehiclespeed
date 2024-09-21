import cv2
import numpy as np
from sklearn.cluster import KMeans

def rgb_to_name(rgb):
    # Define color ranges and names
    color_ranges = {
        'Red': ([0, 0, 100], [80, 80, 255]),
        'Green': ([0, 100, 0], [80, 255, 80]),
        'Blue': ([100, 0, 0], [255, 80, 80]),
        'Yellow': ([0, 100, 100], [80, 255, 255]),
        'Orange': ([0, 60, 100], [80, 150, 255]),
        'Purple': ([80, 0, 80], [255, 80, 255]),
        'Pink': ([180, 0, 180], [255, 100, 255]),
        'Brown': ([0, 0, 0], [100, 100, 100]),
        'White': ([200, 200, 200], [255, 255, 255]),
        'Black': ([0, 0, 0], [50, 50, 50]),
        'Gray': ([80, 80, 80], [200, 200, 200])
    }

    for color_name, (lower, upper) in color_ranges.items():
        if all(lower[i] <= rgb[i] <= upper[i] for i in range(3)):
            return color_name
    return 'Unknown'

def detect_color(image):
    # Resize image to speed up processing
    small_image = cv2.resize(image, (100, 100))
    
    # Convert to RGB color space
    rgb = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = rgb.reshape((-1, 3))
    
    # Perform K-means clustering to find the most dominant color
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)
    
    # Get the RGB values of the cluster center (dominant color)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    
    # Get the name of the dominant color
    color_name = rgb_to_name(dominant_color)
    
    print(f"Dominant RGB: {dominant_color}")
    print(f"Dominant color: {color_name}")
    
    return color_name