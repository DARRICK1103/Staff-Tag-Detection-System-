from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path='models/human_detector.pt'): # yolo12s.pt
        self.model = YOLO(model_path)

    def detect(self, image_rgb):
        return self.model.predict(image_rgb, verbose=False)[0]


class TagDetector:
    def __init__(self, model_path='models/tag_detector.pt', conf_threshold=0.56): # yolo12s.pt
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, person_crop):
        result = self.model.predict(person_crop, verbose=False)[0]
        boxes = [b for b in result.boxes if b.conf[0].item() > self.conf_threshold]
        tag_score = boxes[0].conf[0].item() if boxes else 0.0
        return bool(boxes), tag_score
