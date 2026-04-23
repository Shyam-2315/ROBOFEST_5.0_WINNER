import cv2

from core.vision import VisionSystem
from core.motion_detector import MotionDetector
from core.object_tracker import Tracker
from core.target_lock import TargetLock

from sensors.ultrasonic import UltrasonicSensor
from hardware.servo_controller import ServoController

from utils.fps_counter import FPS
from utils.alert_system import AlertSystem

from ui.dashboard import draw_dashboard

import config


def main() -> None:
    """
    Main loop for ORACLE surveillance system.
    Handles:
    - Real-time camera feed
    - AI object detection + multi-object tracking
    - Target selection, auto target lock, and servo follow (simulation)
    - Motion detection, ultrasonic distance monitoring
    - Smart alert system with siren + on-screen dashboard
    """

    vision = VisionSystem()
    motion = MotionDetector()
    tracker = Tracker()

    ultrasonic = UltrasonicSensor()
    servo = ServoController()

    fps_counter = FPS()

    # Use configurable camera index, and request a smaller resolution for speed
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {config.CAMERA_INDEX}")

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError("Failed to read initial frame from camera")

    h, w, _ = frame.shape

    target_lock = TargetLock(w, h)
    alert_system = AlertSystem(w, h)

    # Cache last detections to avoid running YOLO every single frame
    last_detections = []
    last_tracks = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            # Gracefully exit if camera feed fails
            break

        frame_idx += 1

        # Run heavy detection only every N frames; reuse results in between
        if frame_idx % config.DETECTION_INTERVAL == 0:
            last_detections, last_tracks = vision.detect(frame)

        detections, tracks = last_detections, last_tracks
        motion_flag = motion.detect(frame)

        # Choose highest-priority target (human / vessel, etc.)
        target = tracker.select_target(detections)

        # Ultrasonic distance (cm) for obstacle / collision monitoring
        raw_distance = ultrasonic.get_distance()
        if raw_distance is None:
            distance_value = 9999.0
            distance_display = "N/A"
        else:
            distance_value = float(raw_distance)
            distance_display = int(distance_value)

        # FPS as performance indicator
        fps = fps_counter.update()

        alert_text = None
        target_center = None

        if target:
            x1, y1, x2, y2 = target["bbox"]
            label = target["label"]

            # Auto target lock and servo-based camera follow
            servo_x, servo_y, cx, cy = target_lock.compute((x1, y1, x2, y2))
            servo.move(servo_x, servo_y)

            target_center = (cx, cy)

            # Smart alert system (human rescue, collision, ROI danger zone, suspicious vessel)
            alert_text = alert_system.check_alert(label, cx, cy, distance_value)

        frame = draw_dashboard(
            frame,
            detections,
            tracks,
            fps,
            distance_display,
            motion_flag,
            target_center,
            alert_text,
            (
                alert_system.roi_x1,
                alert_system.roi_y1,
                alert_system.roi_x2,
                alert_system.roi_y2,
            ),
        )

        cv2.imshow("ORACLE SURVEILLANCE SYSTEM", frame)

        # ESC to exit
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()