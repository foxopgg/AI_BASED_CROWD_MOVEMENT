import cv2
import os
import pandas as pd
from datetime import datetime
from src.config import *
from src.analytics import AnalyticsManager

def run_pro_analytics():
    # Path logic for your Linux setup
    
    
    video_path = os.path.expanduser("~/C:\Users\USER\Documents\GitHub\AI_BASED_CROWD_MOVEMENTC:\Users\USER\Documents\GitHub\AI_BASED_CROWD_MOVEMENT")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Video file not found at {video_path}")
        return

    # Metadata for the analytics engine
    w, h = (int(cap.get(3)), int(cap.get(4)))
    analytics = AnalyticsManager(w, h, LINE_POINTS)
    log_data = []

    print("SYSTEM ACTIVE: YOLO11-Large | BoT-SORT Persistence | ROI Zoom enabled")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # --- THE ACCURACY UPGRADE: ROI FOCUS ---
        # Focus the AI on the bottom 80% of the frame where people are clearest
        # This prevents the AI from getting 'distracted' by the sky or ceilings
        roi_frame = frame[int(h*0.2):h, 0:w]

        # 1. Process Heatmap (Visualizes density)
        heatmap_results = analytics.heatmap.process(roi_frame)
        
        # 2. Process Counter (Maintains IDs using custom_tracker.yaml)
        # Passing the heatmap image through allows the visuals to 'stack'
        counter_results = analytics.counter.process(heatmap_results.plot_im)
        final_frame = counter_results.plot_im

        # 3. Intelligence Overlay
        current_count = heatmap_results.total_tracks
        status, color = analytics.classify_density(current_count)
        
        # Professional HUD for Mentor Demo
        cv2.rectangle(final_frame, (0, 0), (450, 70), (0,0,0), -1)
        cv2.putText(final_frame, f"DENSITY: {status}", (20, 30), 1, 1.5, color, 2)
        cv2.putText(final_frame, f"UNIQUE PEOPLE: {current_count}", (20, 60), 1, 1.5, (255,255,255), 2)

        cv2.imshow("Pro-Grade Campus AI Analytics", final_frame)
        
        # Log data for peak-hour graphs
        if len(log_data) % 30 == 0:
            log_data.append({
                "time": datetime.now().strftime("%H:%M:%S"), 
                "count": current_count,
                "in": counter_results.in_count,
                "out": counter_results.out_count
            })

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save to CSV for the 'Data Analytics' part of your project
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(log_data).to_csv(LOG_FILE, index=False)
    print(f"Analytics Exported to {LOG_FILE}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pro_analytics()