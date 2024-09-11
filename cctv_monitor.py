import cv2
import time
import datetime
from ultralytics import YOLO
import yagmail

# Load YOLOv8 model pre-trained on COCO dataset
model = YOLO('yolov8n.pt')  # Use yolov8n.pt for faster detection, change to yolov8x.pt for higher accuracy

# Set your cutoff time (e.g., after 10 PM)
CUTOFF_HOUR = 14  # 10 PM

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
    humans_detected = any([classes[i] == 0 for i in range(len(classes))])  # Check if class 'person' (0) is detected
    return humans_detected, results.render()  # Return detection result and the rendered frame

# Function to capture video feed from the camera
def capture_video():
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or change to an IP camera address
    if not cap.isOpened():
        print("Error: Could not open video feed.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if is_after_cutoff():
            human_present, rendered_frame = detect_human(frame)
            if human_present:
                print("Human detected after cutoff time! Sending alert...")
                #send_alert()

            cv2.imshow('Camera Feed', rendered_frame[0])
        else:
            cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video()
