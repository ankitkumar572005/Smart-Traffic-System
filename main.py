import cv2
import datetime
from tracker import VehicleTracker
from detector import VehicleDetector
from utils import save_to_csv, draw_info

def main():
    video_path = 'sample_traffic.mp4'
    csv_path = 'output/traffic_log.csv'
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        print("Please ensure you run 'wget https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4 -O sample_traffic.mp4'")
        return

    # Video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/annotated_traffic.mp4', fourcc, fps, (width, height))

    # Initialize Engine
    tracker = VehicleTracker(max_age=30)
    detector = VehicleDetector() 
    
    # Define Virtual Line (Horizontal line in lower-middle screen)
    line_start = (0, int(height * 0.6))
    line_end = (width, int(height * 0.6))

    # State
    frame_count = 0
    counted_ids = set()
    total_count = 0
    
    # Store dynamic info (OCR and helmet statuses)
    vehicle_cache = {}

    print("Starting video processing pipeline...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 1. Detection
        detections = detector.detect_vehicles(frame)
        
        # 2. Tracking
        tracks = tracker.update(detections, frame)

        # 3. Handle specific logic per tracked moving vehicle
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb() # xyxy format
            
            # Ensure coordinates are within frame sizes
            x1 = max(0, int(ltrb[0]))
            y1 = max(0, int(ltrb[1]))
            x2 = min(width, int(ltrb[2]))
            y2 = min(height, int(ltrb[3]))
            clamped_ltrb = [x1, y1, x2, y2]
            
            # Centroid y-coordinate
            cy = int((y1 + y2) / 2)
            
            if track_id not in vehicle_cache:
                vehicle_cache[track_id] = {
                    'class_name': track.det_class or 'vehicle',
                    'plate': None,
                    'helmet': None,
                    'last_y': cy
                }
                
            cache = vehicle_cache[track_id]
            
            # --- Event: Line Crossing (Counting) ---
            line_y = line_start[1]
            
            # We check if vehicle center crossed the line bounding downwards
            # A more robust solution uses the exact `check_intersection` per frame delta.
            if cache['last_y'] < line_y and cy >= line_y:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    total_count += 1
                    
                    # Target close up reading
                    if cache['class_name'] == 'motorcycle':
                        cache['helmet'] = detector.check_helmet(frame, clamped_ltrb, track_id)
                    cache['plate'] = detector.read_license_plate(frame, clamped_ltrb)
                    
                    # Record entry
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_to_csv(csv_path, [
                        timestamp, track_id, cache['class_name'], 
                        cache['plate'] or "N/A", cache['helmet'] or "N/A"
                    ])

            cache['last_y'] = cy 
            
            # --- Rendering info ---
            draw_info(frame, track_id, cache['class_name'], clamped_ltrb, 
                      plate_text=cache['plate'], helmet_status=cache['helmet'])
                      
        # Render the line
        cv2.line(frame, line_start, line_end, (0, 255, 255), 2)
        
        # Render the HUD
        cv2.rectangle(frame, (10, 10), (300, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Total Count: {total_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
        # Output frame
        out.write(frame)

        # Logging output to console occasionally so we know it's not frozen
        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count} - Total Counted: {total_count}...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Video output and CSV data saved to 'output/' folder.")

if __name__ == "__main__":
    main()
