import cv2

class MotionDetector:

    def __init__(self):
        self.prev = None

    def detect(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev is None:
            self.prev = gray
            return False

        diff = cv2.absdiff(self.prev, gray)

        motion = diff.sum()

        self.prev = gray

        return motion > 400000