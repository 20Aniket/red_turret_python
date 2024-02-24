import cv2
import numpy as np
from pyfirmata import Arduino, util

prev_center = (0, 0)
min_contour_area = 500

def detect_red(frame):
    global prev_center
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define range of red color in HSV
    lower_red = np.array([0, 100, 255])
    upper_red = np.array([10, 255, 255])
    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    # Define range of red color in HSV (for higher hue values)
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])
    # Threshold the HSV image to get only red colors
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    # Combine the masks
    mask = cv2.bitwise_or(mask1, mask2)  
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour (largest detected red area)
    largest_contour = None
    center_x = 0  # Initialize center_x
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            if largest_contour is None or area > cv2.contourArea(largest_contour):
                largest_contour = contour
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Calculate center of the largest bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        x_threshold= 20
        # Determine direction based on previous center
        if center_x > prev_center[0]+int(x_threshold):
            direction_x = "Right"
        elif center_x < prev_center[0]-int(x_threshold):
            direction_x = "Left"
        else:
            direction_x = ""
        direction_text = direction_x
        cv2.putText(frame, direction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Update previous center
        prev_center = (center_x, center_y)
    return frame, center_x

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def servo_control(arduino_port='/dev/cu.usbmodem101'):
    board = Arduino(arduino_port)
    servopin = 12
    servo = board.get_pin(f"d:{servopin}:s")

    # Open a video capture object
    cap = cv2.VideoCapture(0)
    while True:
        # Read a frame from the video capture object
        ret, frame = cap.read()
        if ret:
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            # Detect red color and draw bounding box
            frame_with_box, center_x = detect_red(frame)
            # Display the resulting frame
            cv2.imshow('Red Detection with Direction', frame_with_box)
            # Map the center_x value to a range of 0 to 180 degrees for the servo
            servo_angle = map_value(center_x, 0, frame.shape[1], 0, 180)
            # Write the servo angle to the servo object
            servo.write(servo_angle)
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Call the servo control function
servo_control()




