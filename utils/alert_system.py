import config
from utils.siren import play_siren


class AlertSystem:

    def __init__(self, frame_width, frame_height):

        self.frame_width = frame_width
        self.frame_height = frame_height

        # Define ROI danger zone
        roi_w = int(frame_width * config.ROI_WIDTH)
        roi_h = int(frame_height * config.ROI_HEIGHT)

        self.roi_x1 = (frame_width - roi_w) // 2
        self.roi_y1 = (frame_height - roi_h) // 2
        self.roi_x2 = self.roi_x1 + roi_w
        self.roi_y2 = self.roi_y1 + roi_h


    def inside_roi(self, cx, cy):

        return (
            self.roi_x1 < cx < self.roi_x2 and
            self.roi_y1 < cy < self.roi_y2
        )


    def check_alert(self, label, cx, cy, distance):

        alert_type = None

        # 1️⃣ Human rescue alert
        if label == config.HUMAN_CLASS:

            alert_type = "HUMAN DETECTED"
            play_siren()

        # 2️⃣ Collision alert
        elif distance < config.COLLISION_DISTANCE:

            alert_type = "COLLISION RISK"
            play_siren()

        # 3️⃣ ROI danger zone alert
        elif self.inside_roi(cx, cy):

            alert_type = "OBJECT IN DANGER ZONE"
            play_siren()

        # 4️⃣ Suspicious vessel alert
        elif label in config.SUSPICIOUS_CLASSES:

            alert_type = "SUSPICIOUS VESSEL"
            play_siren()

        return alert_type