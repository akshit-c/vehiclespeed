import cv2
import numpy as np
from vehicle_detection import detect_vehicles
from number_plate_recognition import recognize_number_plate
from speed_estimation import estimate_speed
from color_detection import detect_color

def main():
    video_path = '/Users/amit/Downloads/h1.mp4'
    
    # Estimate overall speed
    avg_speed = estimate_speed(video_path)
    if avg_speed is not None:
        print(f"Average speed: {avg_speed:.2f} pixels per second")
    else:
        print("Failed to estimate overall speed")

    # Process video frame by frame
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vehicles
        vehicles = detect_vehicles(frame)

        for vehicle in vehicles:
            print(f"Vehicle shape: {vehicle.shape if vehicle is not None else 'None'}")
            print(f"Vehicle type: {type(vehicle)}")
            
            if vehicle is None or vehicle.size == 0:
                print("Error: Vehicle image is empty or None")
                return
            
            # Extract number plate
            number_plate = recognize_number_plate(vehicle)

            # Detect color
            color = detect_color(vehicle)

            # Display results
            cv2.imshow('Vehicle', vehicle)
            if number_plate:
                print(f'Number Plate: {number_plate}, Color: {color}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()