from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    def __init__(self, max_age=30):
        # We restore the 'mobilenet' embedder (very light ~20MB). 
        # Since we removed the massive EasyOCR (~700MB), the app is now safe!
        self.tracker = DeepSort(max_age=max_age, embedder='mobilenet', half=True)

    def update(self, detections, frame):
        """
        Updates the tracker with YOLOv8 detections.
        Expected detections format: list of dicts with 'box', 'confidence', 'class_name'
        """
        bbs = []
        for det in detections:
            x1, y1, x2, y2 = det['box']
            w, h = x2 - x1, y2 - y1
            # DeepSort requires [ [left, top, w, h], confidence, detection_class ]
            bbs.append(([x1, y1, w, h], det['confidence'], det['class_name']))
            
        tracks = self.tracker.update_tracks(bbs, frame=frame)
        return tracks
