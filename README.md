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

---

## Results and Visualizations

### Training Metrics
**Precision, Recall, F1-score, and PR Curves**

![Precision Curve](runs/detect/train3/BoxP_curve.png)  
![Recall Curve](runs/detect/train3/BoxR_curve.png)  
![F1 Curve](runs/detect/train3/BoxF1_curve.png)  
![Precision-Recall Curve](runs/detect/train3/BoxPR_curve.png)  

---

### Confusion Matrices
**Evaluate classification performance**  

![Confusion Matrix](runs/detect/train3/confusion_matrix.png)  
![Normalized Confusion Matrix](runs/detect/train3/confusion_matrix_normalized.png)  

---

### Sample Labels
**Class labels used for training**  

![Labels](runs/detect/train3/labels.jpg)  

---

### Training Batches
**Example images from training batches**  

![Training Batch 0](runs/detect/train3/train_batch0.jpg)  
![Training Batch 1](runs/detect/train3/train_batch1.jpg)  
![Training Batch 2](runs/detect/train3/train_batch2.jpg)  
![Training Batch 6640](runs/detect/train3/train_batch6640.jpg)  
![Training Batch 6641](runs/detect/train3/train_batch6641.jpg)  
![Training Batch 6642](runs/detect/train3/train_batch6642.jpg)  

---

### Validation Results
**Comparison of ground truth and predictions**  

![Validation Batch 0 Labels](runs/detect/train3/val_batch0_labels.jpg)  
![Validation Batch 0 Predictions](runs/detect/train3/val_batch0_pred.jpg)  

![Validation Batch 1 Labels](runs/detect/train3/val_batch1_labels.jpg)  
![Validation Batch 1 Predictions](runs/detect/train3/val_batch1_pred.jpg)  

![Validation Batch 2 Labels](runs/detect/train3/val_batch2_labels.jpg)  
![Validation Batch 2 Predictions](runs/detect/train3/val_batch2_pred.jpg)  

---

### Detection Results
**Overall model predictions**  

![Detection Results](runs/detect/train3/results.png)  

---

## Usage

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
