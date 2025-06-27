import cv2
import os
from staff_detector.detector import PersonDetector, TagDetector
from staff_detector.tracker import PersonTracker

class StaffDetector:
    def __init__(self, video_path, output_path="output"):
        self.person_detector = PersonDetector()
        self.tag_detector = TagDetector()
        self.tracker = PersonTracker()
        self.video_path = video_path
        self.frame_count = 0

        self.cap = cv2.VideoCapture(video_path)
        self.out = cv2.VideoWriter(
            f"{output_path}/output_video.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            30.0,
            (1280, 720)
        )
        self.output_detected_staff_path = f"{output_path}/detected_staff_dataset"
        os.makedirs(self.output_detected_staff_path, exist_ok=True)
        self.tag_score_log = open(f"{output_path}/tag_scores.txt", "w")

    def process_video(self):
        while self.cap.isOpened():
            ret, frame_bgr = self.cap.read()
            if not ret:
                break

            self.frame_count += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.person_detector.detect(frame_rgb)

            for box in results.boxes:
                if int(box.cls[0]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                person_crop = frame_rgb[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tag_found, tag_score = self.tag_detector.detect(person_crop)

                if tag_found:
                    self.tag_score_log.write(f"{self.frame_count},{tag_score:.4f}\n")

                matched, is_staff, pid = self.tracker.update(cx, cy, tag_found, self.frame_count)

                label = "Staff" if is_staff else "Visitor"
                color = (0, 255, 0) if is_staff else (0, 0, 255)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, f'{label} ({tag_score:.2f})', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if is_staff:
                    staff_crop_bgr = frame_bgr[y1:y2, x1:x2]
                    save_path = f"{self.output_detected_staff_path}/frame{self.frame_count}_id{pid}.jpg"
                    cv2.imwrite(save_path, staff_crop_bgr)

            frame_resized = cv2.resize(frame_bgr, (1280, 720))
            self.out.write(frame_resized)
            cv2.imshow("Staff Tag Detection", frame_resized)

            if cv2.waitKey(1) == 27:
                break

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        self.out.release()
        self.tag_score_log.close()
        cv2.destroyAllWindows()
