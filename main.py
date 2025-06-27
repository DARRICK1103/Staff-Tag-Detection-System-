from staff_detector.staff_detector import StaffDetector

if __name__ == "__main__":
    detector = StaffDetector(video_path="videos/sample.mp4")
    detector.process_video()
