class Tracker:

    def select_target(self, detections):

        if not detections:
            return None

        # Priority order for auto target selection:
        # 1. Human (rescue situations)
        # 2. Nearby boats / ships (suspicious / collision risk)
        # 3. Other objects (fish, debris, etc. if model supports classes)
        priority = ["person", "boat", "ship", "fish", "debris"]

        for p in priority:
            for d in detections:
                if d["label"] == p:
                    return d

        return detections[0]