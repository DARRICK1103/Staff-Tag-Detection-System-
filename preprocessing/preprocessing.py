import os
import cv2

# === Path Config ===
IMAGE_FOLDER = "dataset/train/images"
LABEL_FOLDER = "dataset/train/labels"
CROP_OUTPUT_IMAGE_FOLDER = "cropped_dataset/train/images"
CROP_OUTPUT_LABEL_FOLDER = "cropped_dataset/train/labels"

os.makedirs(CROP_OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(CROP_OUTPUT_LABEL_FOLDER, exist_ok=True)

# === Helper Functions ===
def yolo_to_bbox(yolo_box, img_width, img_height):
    """Convert normalized YOLO box to pixel (x1, y1, x2, y2)."""
    x_center, y_center, width, height = yolo_box
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return x1, y1, x2, y2

def bbox_to_yolo(x1, y1, x2, y2, crop_width, crop_height):
    """Convert pixel box to normalized YOLO format in new cropped image."""
    x_center = ((x1 + x2) / 2) / crop_width
    y_center = ((y1 + y2) / 2) / crop_height
    width = (x2 - x1) / crop_width
    height = (y2 - y1) / crop_height
    return x_center, y_center, width, height

# === Process Each File ===
for file_name in os.listdir(IMAGE_FOLDER):

    if not file_name.endswith((".jpg", ".png")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, file_name)
    label_path = os.path.join(LABEL_FOLDER, file_name.replace(".jpg", ".txt").replace(".png", ".txt"))
    if not os.path.exists(label_path):
        continue

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        labels = [line.strip().split() for line in f.readlines()]

    # Step 1: Find the first 'staff' class (class 0)
    staff_boxes = [l for l in labels if l[0] == '0']
    if not staff_boxes:
        continue  # skip if no staff

    staff_box = list(map(float, staff_boxes[0][1:]))  # Just pick the first staff
    sx1, sy1, sx2, sy2 = yolo_to_bbox(staff_box, w, h)

    # Optional: add margin
    margin = 10
    sx1 = max(0, sx1 - margin)
    sy1 = max(0, sy1 - margin)
    sx2 = min(w, sx2 + margin)
    sy2 = min(h, sy2 + margin)

    # Step 2: Crop the image
    cropped_img = img[sy1:sy2, sx1:sx2]
    crop_h, crop_w = cropped_img.shape[:2]

    # Step 3: Adjust labels inside the crop
    new_labels = []
    for label in labels:
        cls_id, x, y, bw, bh = label
        x, y, bw, bh = map(float, (x, y, bw, bh))
        x1, y1, x2, y2 = yolo_to_bbox((x, y, bw, bh), w, h)

        # Check if this box is a staff tag (class 1) and inside the crop
        if cls_id == '1':
            if x1 >= sx1 and y1 >= sy1 and x2 <= sx2 and y2 <= sy2:
                # Adjust to crop coordinates
                cx1 = x1 - sx1
                cy1 = y1 - sy1
                cx2 = x2 - sx1
                cy2 = y2 - sy1
                x_c, y_c, w_c, h_c = bbox_to_yolo(cx1, cy1, cx2, cy2, crop_w, crop_h)
                new_labels.append(f"1 {x_c:.6f} {y_c:.6f} {w_c:.6f} {h_c:.6f}")

    if new_labels:
        # Save cropped image
        output_img_path = os.path.join(CROP_OUTPUT_IMAGE_FOLDER, file_name)
        output_label_path = os.path.join(CROP_OUTPUT_LABEL_FOLDER, file_name.replace(".jpg", ".txt").replace(".png", ".txt"))
        cv2.imwrite(output_img_path, cropped_img)

        # Save label
        with open(output_label_path, "w") as f:
            f.write("\n".join(new_labels))
