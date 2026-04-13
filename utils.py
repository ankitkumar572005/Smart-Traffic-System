import cv2
import csv
import os

def check_intersection(p1, p2, l1, l2):
    """
    Check if line segment p1-p2 intersects with line segment l1-l2.
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, l1, l2) != ccw(p2, l1, l2) and ccw(p1, p2, l1) != ccw(p1, p2, l2)

def save_to_csv(filepath, data):
    """
    Append tracked vehicle data to CSV file.
    data format: [Timestamp, Vehicle ID, Vehicle Type, Number Plate, Helmet Status]
    """
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Vehicle ID', 'Vehicle Type', 'Number Plate', 'Helmet Status'])
        writer.writerow(data)

def draw_info(frame, track_id, class_name, bbox, plate_text=None, helmet_status=None):
    """
    Draw bounding box and annotated text on the frame.
    """
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 0, 255) if helmet_status == 'No Helmet' else (0, 255, 0)
    
    # Draw Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Format the label
    label = f"ID: {track_id} {class_name}"
    
    # Place text above bounding box
    cv2.putText(frame, label, (x1, max(20, y1 - 25)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
    if plate_text:
        cv2.putText(frame, f"Plate: {plate_text}", (x1, max(40, y1 - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
    if helmet_status:
        cv2.putText(frame, helmet_status, (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
