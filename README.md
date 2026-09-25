# Road Damage Detection

This project implements a **road damage detection system** using the YOLOv8 object detection model. The system detects and classifies various types of road damages such as cracks and potholes. It provides bounding boxes, confidence scores, and predicted labels, along with visualizations of training and validation performance.

---

## Project Overview
The **Road Damage Detection** model is trained on labeled road images to automatically identify damage. The system uses YOLOv8 for real-time detection and evaluation. Key features include:  

- Real-time object detection using YOLOv8.  
- Visualization of precision, recall, F1-score, and PR curves.  
- Confusion matrix for evaluation.  
- Sample predictions and validation comparisons.  

---

## Training Configuration
The model is trained using the following configuration (`args.yaml`):

- **Task**: `detect`  
- **Mode**: `train`  
- **Model**: `yolov8n.pt`  
- **Dataset**: `data.yaml`  
- **Epochs**: 50  
- **Batch Size**: 16  
- **Image Size**: 640  
- **Device**: GPU `0`  
- **Optimizer**: Auto  
- **Learning Rate**: `0.01`  
- **IOU Threshold**: `0.7`  
- **Augmentation**: RandAugment, Mosaic, Erasing  

Full configuration is saved in `args.yaml` and can be adjusted for experiments.

1. Clone the repository and navigate to the project folder.  
2. Install dependencies: PyTorch, YOLOv8, OpenCV, Matplotlib, Pandas, Seaborn.  
3. Configure your dataset path in `data.yaml`.  
4. Train the model:

```bash
yolo train model=yolov8n.pt data=data.yaml epochs=50 batch=16 imgsz=640
```
5. Evaluate the model:
```bash
yolo val model=runs/detect/train3/weights/best.pt data=data.yaml
```
6. Run inference on new images or videos:
```bash
yolo predict model=runs/detect/train3/weights/best.pt source="path/to/video.mp4"
```

## Requirements

- Python 3.9+
- PyTorch 2.x
- Ultralytics YOLOv8
- OpenCV
- Matplotlib, Pandas, Seaborn
