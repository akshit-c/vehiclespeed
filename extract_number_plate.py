import pytesseract
import cv2

cap = cv2.VideoCapture('/Users/amit/Downloads/2034115-hd_1920_1080_30fps.mp4')  # Use 0 for default camera, or provide a video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the bounding box coordinates from the previous step
    (x, y, w, h) = cv2.boundingRect(frame)

    # Extract the number plate region
    roi = frame[y:y+h, x:x+w]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance the number plate
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use Tesseract OCR to recognize the characters
    text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 11')

    # Print the recognized text
    print(text)

    # Display the output
    cv2.imshow('Number Plate Recognition', roi)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()