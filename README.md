# Smart Traffic Monitoring System

An AI/ML-based vehicle detection, tracking, counting, and license-plate/helmet identification script using YOLOv8, DeepSORT, and EasyOCR.

## Features Included
1. **Vehicle Detection:** Ignores pedestrians, strictly detects `car`, `motorcycle`, `bus`, `truck` seamlessly using YOLOv8.
2. **Object Tracking:** Uses `deep-sort-realtime` to stably assign a unique ID across frames.
3. **Line Crossing & Counting:** Contains an internal counting logic. When a tracked centroid crosses the virtual line on the screen, the vehicle is counted exactly once.
4. **License Plate Reading (ANPR):** Automatically crops the detected vehicle bounding box, runs thresholding and passes it through `easyocr` to detect number plates. (Fallback logic implemented if custom weights aren't supplied).
5. **Helmet Detection:** Specifically targets `motorcycles` when they cross the capture line.
6. **Data Storage:** Exports events iteratively to `output/traffic_log.csv`. 
7. **Recording:** Records annotated footage matching the input video specs automatically to `output/annotated_traffic.mp4`.

## Project Structure
```text
smart_traffic_system/
├── requirements.txt
├── README.md
├── main.py                - System pipeline architecture
├── detector.py            - Inference logic for YOLOv8 and EasyOCR
├── tracker.py             - DeepSORT wrapper module 
├── utils.py               - Vector math and OpenCV rendering tools
├── models/                - Required. Pretrained model files (.pt)
└── output/                - Stores output annotated video and CSV
```

## Setup Instructions

1. **Environment Setup**
    Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Get weights & a video:**
    You must provide your own target `.mp4` file explicitly in `main.py`, or simply place a sample target traffic footage and name it `sample_traffic.mp4`.
    To use the sample video:
    ```bash
    curl -L -o sample_traffic.mp4 "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
    ```

4. **Run the tool:**
    ```bash
    python main.py
    ```

## Custom Domain Tracking
By default, the script dynamically pulls `yolov8n.pt` and utilizes mocked logic combined with OCR fallback methods for plates and helmets logic respectively if they are missing true training models.
If you have trained models for Plate Detection and Helmet detection (like from Roboflow Universe), place the `.pt` files in `models/` and link them directly during initialization:
`detector = VehicleDetector(plate_model_path="models/plate.pt", helmet_model_path="models/helmet.pt")` 
