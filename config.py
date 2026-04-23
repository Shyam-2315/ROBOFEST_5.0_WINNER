CAMERA_INDEX = 0

MODEL_PATH = "models/yolov8n.pt"

# --- Runtime / performance tuning ---

# Target camera resolution (smaller = faster)
# Aggressively reduced to help low-power laptops.
FRAME_WIDTH = 480
FRAME_HEIGHT = 360

# YOLO inference image size (must be multiple of 32, 320–416 is good for speed)
# Lower = faster. 320 is chosen for maximum responsiveness.
DETECTION_IMG_SIZE = 320

# Run full detection every N frames and reuse results in between
# 1 = detect every frame (slower, smoother), higher = faster
DETECTION_INTERVAL = 4

# Object classes considered threats
HUMAN_CLASS = "person"

SUSPICIOUS_CLASSES = [
    "boat",
    "ship"
]

# Distance threshold (cm)
COLLISION_DISTANCE = 100

# ROI danger zone size (percentage of frame)
ROI_WIDTH = 0.4
ROI_HEIGHT = 0.4