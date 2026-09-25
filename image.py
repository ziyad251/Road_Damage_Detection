import glob
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Path to folder containing test images
image_folder = "test_images"   # change this to your folder path

# Get all jpg files
image_files = glob.glob(f"{image_folder}/*.jpg")

# Loop through all images
for img_path in image_files:
    results = model(img_path)
    res = results[0]
    res.save(filename=img_path.replace(".jpg", "_detected.jpg"))
    print(f"Processed: {img_path}")
