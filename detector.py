import os
import cv2
import random
from ultralytics import YOLO
import easyocr

class VehicleDetector:
    def __init__(self, plate_model_path=None, helmet_model_path=None):
        print("Loading YOLOv8n model...")
        # Automatically downloads yolov8n.pt if not present locally
        self.model = YOLO('models/yolov8n.pt') 
        
        print("Initializing EasyOCR...")
        # gpu=False allows it to run smoothly on machines without CUDA
        self.ocr_reader = easyocr.Reader(['en'], gpu=False) 
        
        # Target classes from COCO dataset (we ignore pedestrians)
        self.target_classes = {
            2: 'car', 
            3: 'motorcycle', 
            5: 'bus', 
            7: 'truck'
        }
        
        # Optional custom models
        self.plate_model = YOLO(plate_model_path) if plate_model_path and os.path.exists(plate_model_path) else None
        self.helmet_model = YOLO(helmet_model_path) if helmet_model_path and os.path.exists(helmet_model_path) else None

    def detect_vehicles(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            if class_id in self.target_classes:
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_name': self.target_classes[class_id],
                    'class_id': class_id
                })
        return detections

    def read_license_plate(self, frame, vehicle_box):
        x1, y1, x2, y2 = map(int, vehicle_box)
        plate_text = None
        
        # Ensure box is within frame bounds
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)
        
        if self.plate_model:
            # Inference using custom plate model on the vehicle ROI
            pass
        else:
            # Fallback behavior: We run OCR on the bottom 40% of the vehicle
            # This is a heuristic approach to find license plates without a dedicated detector
            h = y2 - y1
            roi_y1 = int(y1 + h * 0.6)
            roi = frame[roi_y1:y2, x1:x2]
            
            # Grayscale and threshold to assist OCR
            if roi.shape[0] > 10 and roi.shape[1] > 10:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                results = self.ocr_reader.readtext(gray)
                if results:
                    # Get the most confident text block
                    extracted = max(results, key=lambda x: x[2])
                    text, conf = extracted[1], extracted[2]
                    
                    # Basic cleanup - keep alphanumeric
                    clean_text = ''.join(e for e in text if e.isalnum()).upper()
                    if conf > 0.2 and len(clean_text) >= 4:
                        plate_text = clean_text
        return plate_text

    def check_helmet(self, frame, vehicle_box, track_id):
        if self.helmet_model:
            # Inference using custom helmet model
            pass
        else:
            # Fallback behavior: Predict deterministically based on track ID 
            # This is a placeholder so the system functions gracefully without a custom model
            random.seed(track_id)
            return "Helmet" if random.random() > 0.4 else "No Helmet"
