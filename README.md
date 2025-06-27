# Staff Tag Detection System

This project performs automated detection of staff members in video footage using a two-stage YOLOv8-based pipeline. It first detects humans, then classifies whether a staff tag is present via a second model.

## 📌 Features

- YOLOv8-based person detection (`yolo12s.pt`)
- Staff tag detection via secondary model (`best.pt`)
- Confidence-based tag verification
- Historical tracking to reduce false positives
- Staff image cropping and saving to folder
- Video output with annotations
- Dataset preprocessing and visualization

---

## 🗃 Dataset Preparation & Verification

### Dataset Source
This project uses a custom YOLOv8-compatible dataset originally from Roboflow:  
🔗 [Staff Tag Detection on Roboflow](https://universe.roboflow.com/shanahan-suresh1-gmail-com/staff-tag-detection)
