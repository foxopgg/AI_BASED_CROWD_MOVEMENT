import cv2
from ultralytics import solutions

class AnalyticsManager:
    def __init__(self, width, height, line_pts):
        # Layer 1: Heatmap for Bottleneck Detection
        self.heatmap = solutions.Heatmap(
            model="yolo11l.pt",  # Large model for better crowd resolution
            colormap=cv2.COLORMAP_JET,
            conf=0.30,
            classes=[0]         # STRICTLY Persons only
        )
        
        # Layer 2: Object Counter for Flow Tracking
        self.counter = solutions.ObjectCounter(
            region=line_pts, 
            model="yolo11l.pt",
            classes=[0],        # STRICTLY Persons only
            show=False,
            tracker="custom_tracker.yaml" 
        )

    def classify_density(self, count):
        # Actionable insights for your campus demo
        if count < 10: return "LOW", (0, 255, 0)
        elif count < 25: return "MEDIUM", (0, 255, 255)
        else: return "HIGH", (0, 0, 255)