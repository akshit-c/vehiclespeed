Speed Vision (ANPR System - Computer Vision, OpenCV, Python)
OBJECTIVE
Design a desktop application that processes video streams to detect vehicles, identify and recognize number plates using Optical Character Recognition (OCR), and estimate vehicle speed. Additionally, provide a feature to detect vehicle color for bonus points.
Tech Stack
Programming Language: Python
Libraries: OpenCV for video processing, Tesseract for OCR to read number plates, and Numpy/Scipy for calculating speed.
Optional: Use YOLO or SSD models via TensorFlow or PyTorch for more accurate vehicle detection.
Key Features:
	1	﻿﻿﻿Real-Time Video Processing: Capture live or pre-recorded video streams to detect moving vehicles.
	2	﻿﻿﻿Vehicle Detection: Use OpenCV or a deep learning model (e.g., YOLOv5) to detect vehicles in the video frame.
	3	﻿﻿﻿Number Plate Recognition: Use Tesseract OCR to extract characters from the vehicle's number plate.
	4	﻿﻿﻿Speed Estimation: Calculate vehicle speed by analyzing its movement across frames. Track the vehicle's position over time to determine how far it travels between frames, and compute speed accordingly.
	5	﻿﻿﻿Color Detection (Bonus): Implement a color detection algorithm to identify the vehicle's color based on pixel
analysis.
Detailed Workflow:
Step 1: Use OpenCV to capture and process the video input.
Step 2: Apply a pre-trained vehicle detection model (e.g., YOLO) to identify vehicles within the video frames.
Step 3: Extract the number plate from the detected vehicle and use Tesseract OCR to recognize the characters.
Step 4: Estimate the speed by measuring the vehicle's movement across consecutive frames and calculating the distance traveled over time.
Step 5: Use color analysis to detect the vehicle's color by analyzing the dominant color in the vehicle's bounding box.
Evaluation Criteria:
Detection Accuracy: How accurately does the system detect vehicles and recognize number plates?
Speed Calculation: How accurately is the vehicle's speed estimated based on frame analysis?
Bonus: How effectively does the system identify the vehicle's color?
Overall Efficiency: Can the system process video streams in real-time without significant lag or errors?
