import cv2
import datetime
import numpy as np
import gc
from tracker import VehicleTracker
from detector import VehicleDetector
from utils import save_to_csv, draw_info

def process_video(video_path, output_mp4, csv_path, detector, progress_bar=None, status_text=None):
    from tracker import VehicleTracker
    from utils import save_to_csv, draw_info

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # FORCE 360p RESOLUTION for RAM protection (Rescue Mode)
    # This reduces memory usage aggressively to survive 1GB RAM
    target_width = 640
    target_height = 360
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (target_width, target_height))
    
    # Update dimensions for the rest of the script
    width, height = target_width, target_height

    # Initialize Engine
    tracker = VehicleTracker(max_age=30)
    # Detector passed from app.py cached resource    
    # Define Virtual Lines for robust counting across any camera angle
    line_y = int(height * 0.6)
    line_x = int(width * 0.5)
    line_start_y = (0, line_y)
    line_end_y = (width, line_y)
    line_start_x = (line_x, 0)
    line_end_x = (line_x, height)

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
        
        # Immediate resize to 480p to save memory
        frame = cv2.resize(frame, (target_width, target_height))
        
        # 1. Detection (PRODUCTION OPTIMIZATION: Skip 2 frames, Process 1)
        # This gives a 3x speed boost while tracking remains smooth
        # 1. Run detection on every frame for reliable tracking
        scale = 640 / max(width, height)
        ai_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        detections_raw = detector.detect_vehicles(ai_frame)
        
        # Rescale boxes back to original size
        detections = []
        for det in detections_raw:
            det['box'] = [b / scale for b in det['box']]
            detections.append(det)
        
        # 2. Tracking (Always run to maintain smooth path)
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
            
            # Centroids
            cy = int((y1 + y2) / 2)
            cx = int((x1 + x2) / 2)
            
            if track_id not in vehicle_cache:
                vehicle_cache[track_id] = {
                    'class_name': track.det_class or 'vehicle',
                    'plate': None,
                    'helmet': None,
                    'last_y': cy,
                    'last_x': cx
                }
                
            cache = vehicle_cache[track_id]
            
            # --- Continuous processing zone ---
            
            # SPEED OPTIMIZATION: Only run expensive OCR/Helmet logic every 10 frames
            # and STOP once we find a clear result
            zone_tolerance = int(height * 0.45) 
            if abs(cy - line_y) < zone_tolerance and frame_count % 10 == 0:
                if cache['plate'] is None:
                    plate = detector.read_license_plate(frame, clamped_ltrb)
                    if plate: cache['plate'] = plate
                        
                if cache['class_name'] in ['motorcycle', 'bicycle']:
                    # Only stop if we have a definitive "Helmet" or "No Helmet"
                    if cache['helmet'] is None or cache['helmet'] == "Checking...":
                        status = detector.check_helmet(frame, clamped_ltrb, track_id)
                        if status != "Checking...":
                            cache['helmet'] = status

            # --- Event: Line Crossing (Counting) ---
            crossed_downward = cache['last_y'] < line_y and cy >= line_y
            crossed_upward = cache['last_y'] > line_y and cy <= line_y
            crossed_right = cache['last_x'] < line_x and cx >= line_x
            crossed_left = cache['last_x'] > line_x and cx <= line_x
            
            if crossed_downward or crossed_upward or crossed_right or crossed_left:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    total_count += 1
                    
                    # Record entry at the moment of crossing
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_to_csv(csv_path, [
                        timestamp, track_id, cache['class_name'], 
                        cache['plate'] or "N/A", cache['helmet'] or "N/A"
                    ])

            cache['last_y'] = cy 
            cache['last_x'] = cx
            
            # --- Rendering info ---
            draw_info(frame, track_id, cache['class_name'], clamped_ltrb, 
                      plate_text=cache['plate'], helmet_status=cache['helmet'])
                      
        # Render the lines
        cv2.line(frame, line_start_y, line_end_y, (0, 255, 255), 2)
        cv2.line(frame, line_start_x, line_end_x, (0, 255, 255), 2)
        
        # Render the HUD
        cv2.rectangle(frame, (10, 10), (300, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Total Count: {total_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
        # Output frame
        out.write(frame)

        if progress_bar and total_frames > 0:
            if frame_count % 5 == 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
        if status_text and frame_count % 30 == 0:
            status_text.text(f"Processed frame {frame_count}/{total_frames} - Total Counted: {total_count}...")
            # Collect garbage every 30 frames to keep RAM low
            gc.collect()
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    if progress_bar:
        progress_bar.progress(1.0)
    if status_text:
        status_text.text("Processing complete!")
        
    return True

if __name__ == "__main__":
    detector = VehicleDetector()
    process_video('sample_traffic.mp4', 'output/annotated_traffic.mp4', 'output/traffic_log.csv', detector)
