from ultralytics import YOLO
import torch
import config


class VisionSystem:

    def __init__(self):
        # Automatically pick GPU if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(config.MODEL_PATH)
        # Move model to the selected device when supported
        try:
            self.model.to(self.device)
        except Exception:
            self.device = None

    def detect(self, frame):
        """
        Pure detection (NO tracking IDs) for maximum speed.
        Returns a list of detections and an empty tracks list.
        """

        kwargs = {
            "imgsz": config.DETECTION_IMG_SIZE,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        # Ultralytics models are callable; take first result
        results = self.model(frame, **kwargs)[0]

        detections = []

        if results.boxes is None:
            return detections, []

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = self.model.names[int(classes[i])]

            detections.append(
                {
                    "label": label,
                    "bbox": (x1, y1, x2, y2),
                }
            )

        # Second value kept for API compatibility but is always empty (no tracking)
        return detections, []