import os
import random
import shutil

# Path to your flat dataset folder
SOURCE_DIR = "archive"   # change this
DEST_DIR = "dataset"     # new organized folder

# Split ratio
train_ratio = 0.8  # 80% train, 20% val

# Create YOLO structure
for subdir in [
    "images/train", "images/val",
    "labels/train", "labels/val"
]:
    os.makedirs(os.path.join(DEST_DIR, subdir), exist_ok=True)      

# Get all image files
images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.shuffle(images)
split_index = int(len(images) * train_ratio)
train_imgs = images[:split_index]
val_imgs = images[split_index:]

def copy_pair(img_list, subset):
    for img_file in img_list:
        txt_file = os.path.splitext(img_file)[0] + ".txt"
        img_src = os.path.join(SOURCE_DIR, img_file)
        txt_src = os.path.join(SOURCE_DIR, txt_file)
        img_dest = os.path.join(DEST_DIR, f"images/{subset}", img_file)
        txt_dest = os.path.join(DEST_DIR, f"labels/{subset}", txt_file)

        if os.path.exists(txt_src):  # copy only if pair exists
            shutil.copy(img_src, img_dest)
            shutil.copy(txt_src, txt_dest)

copy_pair(train_imgs, "train")
copy_pair(val_imgs, "val")

print(f"✅ Dataset prepared successfully!")
print(f"Train images: {len(train_imgs)}, Val images: {len(val_imgs)}")
print(f"Saved to: {DEST_DIR}")
