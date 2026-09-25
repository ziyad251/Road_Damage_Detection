import glob
import cv2
from ultralytics import YOLO
import os

# Load trained model
model = YOLO("best.pt")

# Path to folder containing videos
video_folder = "test_video"  # change path as needed

# Output folder for detected videos
output_folder = os.path.join(video_folder, "output_detected")
os.makedirs(output_folder, exist_ok=True)

# Get all video files (mp4, avi, mov)
video_files = glob.glob(f"{video_folder}/*.mp4") + \
               glob.glob(f"{video_folder}/*.avi") + \
               glob.glob(f"{video_folder}/*.mov")

for video_path in video_files:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join(
        output_folder,
        os.path.basename(video_path).replace(".mp4", "_detected.mp4")
    )

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Processing: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, stream=True)
        for r in results:
            annotated_frame = r.plot()
            out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Saved: {output_path}")

print("✅ All videos processed and saved in:", output_folder)
