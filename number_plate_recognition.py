import pytesseract
import cv2
import numpy as np

def recognize_number_plate(frame, vehicle):
    x1, y1, x2, y2, _, _ = vehicle
    
    # Extract the license plate region from the image
    plate_region = frame[int(y1):int(y2), int(x1):int(x2)]
    
    # Preprocess the image
    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours and keep only the largest ones
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    plate_roi = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 2.5 < aspect_ratio < 5.5:  # Typical aspect ratio for license plates
            roi = opening[y:y+h, x:x+w]
            plate_roi = (x1 + x, y1 + y, x1 + x + w, y1 + y + h)
            break
    else:
        roi = opening
    
    # Perform OCR using pytesseract
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(roi, config=config)
    
    # Remove any whitespace and newline characters
    cleaned_text = ''.join(text.split())
    
    return cleaned_text if cleaned_text else "Unknown", plate_roi