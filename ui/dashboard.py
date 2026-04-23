import cv2


def draw_dashboard(frame, detections, tracks, fps, distance,
                   motion_flag, target_center, alert_text, roi):

    # Draw detection boxes
    for d in detections:

        x1, y1, x2, y2 = d["bbox"]
        label = d["label"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Tracking removed for performance; 'tracks' is unused and kept only for API compatibility

    # Target red dot
    if target_center:

        cx, cy = target_center

        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    # ROI (danger zone) is still used internally for alerts,
    # but we no longer draw the box on the frame to keep UI clean.

    # System info
    cv2.putText(frame, f"FPS: {fps}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Distance: {distance} cm", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    motion_text = "Motion Detected" if motion_flag else "No Motion"

    cv2.putText(frame, motion_text, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Alert message
    if alert_text:

        cv2.putText(frame, alert_text,
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 3)

    return frame