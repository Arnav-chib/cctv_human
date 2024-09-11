import cv2
import torch
import time
from ultralytics import YOLO
import numpy as np
import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
import yagmail

# Load YOLOv8 model (using GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = YOLO('yolov8x.pt').to(device)  # Using YOLOv8 medium model for better accuracy

# DeepSORT tracker setup
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, max_iou_distance=0.7)

# Set cutoff time and cooldown period
CUTOFF_HOUR = 20  # 8 PM
COOLDOWN_PERIOD = 300  # 5 minutes
last_alert_time = 0

# Function to check if it's after the cutoff time
def is_after_cutoff():
    current_time = datetime.datetime.now().time()
    return current_time.hour >= CUTOFF_HOUR

# Function to send an alert
# def send_alert():
#     receiver = "recipient@example.com"  # Replace with actual recipient email
#     body = "Human movement detected after cutoff time!"
#     yag = yagmail.SMTP("youremail@gmail.com", "yourpassword")  # Replace with your credentials
#     yag.send(
#         to=receiver,
#         subject="Alert: Human Detected",
#         contents=body,
#     )
#     print("Alert sent!")

# Function to process detection results and track humans
def process_detections(frame, results):
    # Extract bounding boxes and classes from YOLOv8 results
    detections = []
    for r in results[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        class_id = int(r.cls[0])
        confidence = float(r.conf[0])
        
        if class_id == 0:  # Class 0 corresponds to 'person' in COCO
            detections.append(([x1, y1, x2, y2], confidence, class_id))
    
    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracked objects on the frame
    human_detected = False
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        bbox = track.to_ltrb()  # Get bounding box

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Person {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        human_detected = True  # A human has been detected in this frame
    
    return human_detected, frame

# Video capture function
def capture_video():
    global last_alert_time
    cap = cv2.VideoCapture(0)  # Change to the appropriate camera index or URL for IP cameras

    if not cap.isOpened():
        print("Error: Could not open video feed.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        if is_after_cutoff():
            results = model(frame)  # Run YOLOv8 inference on GPU
            human_present, processed_frame = process_detections(frame, results)

            if human_present:
                if current_time - last_alert_time > COOLDOWN_PERIOD:
                    print("Human detected after cutoff time! Sending alert...")
                    #send_alert()
                    last_alert_time = current_time  # Update last alert time

            cv2.imshow('Camera Feed', processed_frame)
        else:
            cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video()
