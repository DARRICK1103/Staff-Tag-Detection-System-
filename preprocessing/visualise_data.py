import os
import cv2

# === Path Configuration ===
IMAGE_FOLDER = "cropped_dataset/train/images"
LABEL_FOLDER = "cropped_dataset/train/labels"
CLASS_NAMES = ["staff", "staff_tag"]  # adjust if needed

# Optional: output folder to save visualizations
OUTPUT_FOLDER = "visualized_dataset"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Visualization Function ===
def visualize_yolo_annotations(image_path, label_path, class_names):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return
    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"No label for {image_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        cls_id, x_center, y_center, box_w, box_h = map(float, line.strip().split())
        cls_id = int(cls_id)

        # Convert YOLO (normalized) to pixel coordinates
        x_center *= w
        y_center *= h
        box_w *= w
        box_h *= h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, class_names[cls_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show image
    cv2.imshow("Label Visualization", img)
    key = cv2.waitKey(0)

    # Press 's' to save
    if key == ord('s'):
        out_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
        cv2.imwrite(out_path, img)

    cv2.destroyAllWindows()

# === Loop Through Images ===
for filename in os.listdir(IMAGE_FOLDER):
    if not filename.endswith((".jpg", ".png")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)
    label_path = os.path.join(LABEL_FOLDER, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

    visualize_yolo_annotations(image_path, label_path, CLASS_NAMES)
