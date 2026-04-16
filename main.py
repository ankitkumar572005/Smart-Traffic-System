import cv2
import datetime
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

    # Video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))

    # Initialize Engine
    tracker = VehicleTracker(max_age=30)
    # Detector passed from app.py cached resource    
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
        
        # 1. Detection (PRODUCTION OPTIMIZATION: Skip 2 frames, Process 1)
        # This gives a 3x speed boost while tracking remains smooth
        if frame_count % 3 == 0:
            scale = 640 / max(width, height)
            ai_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            detections_raw = detector.detect_vehicles(ai_frame)
            
            # Rescale boxes back to original size
            detections = []
            for det in detections_raw:
                det['box'] = [b / scale for b in det['box']]
                detections.append(det)
        else:
            detections = [] # Tracker predicts for the missing frames
        
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
            
            # --- Continuous processing zone ---
            line_y = line_start[1]
            
            # SPEED OPTIMIZATION: Only run expensive OCR/Helmet logic every 10 frames
            # and STOP once we find a clear result
            zone_tolerance = int(height * 0.45) 
            if abs(cy - line_y) < zone_tolerance and frame_count % 10 == 0:
                if cache['plate'] is None:
                    plate = detector.read_license_plate(frame, clamped_ltrb)
                    if plate: cache['plate'] = plate
                        
                if cache['class_name'] == 'motorcycle':
                    # Only stop if we have a definitive "Helmet" or "No Helmet"
                    if cache['helmet'] is None or cache['helmet'] == "Checking...":
                        status = detector.check_helmet(frame, clamped_ltrb, track_id)
                        if status != "Checking...":
                            cache['helmet'] = status

            # --- Event: Line Crossing (Counting) ---
            crossed_downward = cache['last_y'] < line_y and cy >= line_y
            crossed_upward = cache['last_y'] > line_y and cy <= line_y
            
            if crossed_downward or crossed_upward:
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

        if progress_bar and total_frames > 0:
            if frame_count % 5 == 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
        if status_text and frame_count % 30 == 0:
            status_text.text(f"Processed frame {frame_count}/{total_frames} - Total Counted: {total_count}...")
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    if progress_bar:
        progress_bar.progress(1.0)
    if status_text:
        status_text.text("Processing complete!")
        
    return True

if __name__ == "__main__":
    process_video('sample_traffic.mp4', 'output/annotated_traffic.mp4', 'output/traffic_log.csv')
