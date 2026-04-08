import cv2
from ultralytics import YOLO

class CrowdDetector:
    def __init__(self, model_path, imgsz):
        self.model = YOLO(model_path)
        self.imgsz = imgsz

    def get_tracks(self, frame):
        # class 0 is 'person' in COCO dataset
        results = self.model.track(
            frame, 
            persist=True, 
            classes=[0], 
            imgsz=self.imgsz, 
            verbose=False,
            half=True  # Enables FP16 for 3050 VRAM efficiency
        )
        return results