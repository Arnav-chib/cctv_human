import cv2
import time
import datetime
from ultralytics import YOLO
import yagmail

# Load YOLOv8 model pre-trained on COCO dataset
model = YOLO('yolov8x.pt')  # Use yolov8n.pt for faster detection, change to yolov8x.pt for higher accuracy

# Set your cutoff time (e.g., after 10 PM)
CUTOFF_HOUR = 20  # 10 PM

# Set cooldown period (e.g., 5 minutes)
COOLDOWN_PERIOD = 300  # in seconds (5 minutes)

# Last alert timestamp
last_alert_time = 0

# Function to check if the current time is after the cutoff time
def is_after_cutoff():
    current_time = datetime.datetime.now().time()
    return current_time.hour >= CUTOFF_HOUR

# Function to send an alert email when a human is detected after cutoff time
# def send_alert():
#     receiver = "recipient@example.com"  # Replace with your email
#     body = "Human movement detected after cutoff time!"
#     yag = yagmail.SMTP("youremail@gmail.com", "yourpassword")  # Replace with your email credentials
#     yag.send(
#         to=receiver,
#         subject="Alert: Human Detected",
#         contents=body,
#     )
#     print("Alert sent!")

# Function to detect humans in the video feed using YOLOv8
def detect_human(frame):
    results = model(frame)
    detections = results[0].boxes.xyxy  # Get bounding boxes
    classes = results[0].boxes.cls  # Get classes of detected objects
    humans_detected = False
    
    for i in range(len(classes)):
        if classes[i] == 0:  # Class 0 corresponds to 'person'
            humans_detected = True
            x1, y1, x2, y2 = map(int, detections[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return humans_detected, frame  # Return detection result and the modified frame with bounding boxes

# Function to capture video feed from the camera
def capture_video():
    global last_alert_time

    cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or change to an IP camera address
    if not cap.isOpened():
        print("Error: Could not open video feed.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        if is_after_cutoff():
            human_present, rendered_frame = detect_human(frame)
            if human_present:
                # Check if the cooldown period has passed since the last alert
                if current_time - last_alert_time > COOLDOWN_PERIOD:
                    print("Human detected after cutoff time! Sending alert...")
                    # send_alert()
                    last_alert_time = current_time  # Update the last alert time

            cv2.imshow('Camera Feed', rendered_frame)
        else:
            cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video()
