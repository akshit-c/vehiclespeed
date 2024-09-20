import cv2
import pytesseract

def recognize_number_plate(vehicle):
    # Pre-process image
    gray = cv2.cvtColor(vehicle, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find number plate contour
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        if area > 1000 and aspect_ratio > 2:
            # Extract number plate
            number_plate = vehicle[y:y+h, x:x+w]

            # Recognize text using Tesseract OCR
            text = pytesseract.image_to_string(number_plate, config='--psm 11')

            return text

    return None