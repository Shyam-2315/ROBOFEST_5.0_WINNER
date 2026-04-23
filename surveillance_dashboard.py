"""
ORACLE Autonomous Surveillance Dashboard
A modern, real-time surveillance dashboard UI built with PySide6.

Features:
- Live camera feed with object detection overlay
- Ultrasonic sensor distance readings
- Radar visualization with object tracking
- Dark maritime-themed UI
- Fully simulated data (no backend required)
"""

import sys
import random
import math
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QGridLayout
)
from PySide6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QFont, QImage, QBrush,
    QLinearGradient
)
from PySide6.QtCore import Qt, QTimer, QSize, QPoint, QRect, QThread, Signal
import numpy as np
import cv2

# Import core ORACLE surveillance pipeline (same as main.py)
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


class CameraFeedWidget(QFrame):
    """Widget for displaying real camera feed with ORACLE detection overlays."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet("border: 2px solid #1a4d5c; border-radius: 8px; background-color: #0a1f2e;")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)
        
        # Title
        title = QLabel("Live Camera Feed - ORACLE System")
        title.setStyleSheet("color: #00d9ff; font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setMinimumHeight(400)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #050f19;")
        layout.addWidget(self.video_label)
        
        # Core ORACLE components (same logic as main.py)
        self.vision = VisionSystem()
        self.motion = MotionDetector()
        self.tracker = Tracker()
        self.ultrasonic = UltrasonicSensor()
        self.servo = ServoController()
        self.fps_counter = FPS()

        # Camera configuration
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        ret, frame = self.cap.read()
        if not ret or frame is None:
            frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)

        h, w, _ = frame.shape
        self.target_lock = TargetLock(w, h)
        self.alert_system = AlertSystem(w, h)

        # Detection cache to reduce heavy YOLO calls
        self.last_detections = []
        self.last_tracks = []
        self.frame_idx = 0
        
        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS UI refresh
    
    def update_frame(self):
        """Update the camera feed with real ORACLE detections."""

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return
        else:
            return

        self.frame_idx += 1

        # Run YOLO only every Nth frame
        if self.frame_idx % config.DETECTION_INTERVAL == 0:
            self.last_detections, self.last_tracks = self.vision.detect(frame)

        detections, tracks = self.last_detections, self.last_tracks

        motion_flag = self.motion.detect(frame)
        target = self.tracker.select_target(detections)
        distance = self.ultrasonic.get_distance()
        fps = self.fps_counter.update()

        alert_text = None
        target_center = None

        if target:
            x1, y1, x2, y2 = target["bbox"]
            label = target["label"]

            servo_x, servo_y, cx, cy = self.target_lock.compute((x1, y1, x2, y2))
            self.servo.move(servo_x, servo_y)

            target_center = (cx, cy)
            alert_text = self.alert_system.check_alert(label, cx, cy, distance)
        
        # Use shared OpenCV dashboard renderer (same as main.py)
        frame = draw_dashboard(
            frame,
            detections,
            tracks,
            fps,
            distance,
            motion_flag,
            target_center,
            alert_text,
            (
                self.alert_system.roi_x1,
                self.alert_system.roi_y1,
                self.alert_system.roi_x2,
                self.alert_system.roi_y2,
            ),
        )
        
        # Convert to QPixmap and display in Qt label
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)


class UltrasonicWidget(QFrame):
    """Widget for displaying ultrasonic sensor readings."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet("border: 2px solid #1a4d5c; border-radius: 8px; background-color: #0a1f2e;")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Ultrasonic Distance Readings")
        title.setStyleSheet("color: #00d9ff; font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        # Sensor values
        self.sensor_data = {
            'Front': {'distance': 2.5, 'widget': None},
            'Left': {'distance': 1.8, 'widget': None},
            'Right': {'distance': 3.2, 'widget': None}
        }
        
        grid = QGridLayout()
        grid.setSpacing(10)
        
        for idx, (name, data) in enumerate(self.sensor_data.items()):
            sensor_frame = self.create_sensor_display(name, data['distance'])
            data['widget'] = sensor_frame
            grid.addWidget(sensor_frame, idx, 0)
        
        layout.addLayout(grid)
        layout.addStretch()
        
        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_sensors)
        self.timer.start(1000)  # Update every second
    
    def create_sensor_display(self, name, distance):
        """Create a sensor display widget."""
        frame = QFrame()
        frame.setStyleSheet("border: 1px solid #1a4d5c; border-radius: 6px; background-color: #051820;")
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Sensor name
        name_label = QLabel(name)
        name_label.setStyleSheet("color: #00d9ff; font-weight: bold; font-size: 11px;")
        layout.addWidget(name_label)
        
        # Distance display (digital style)
        distance_label = QLabel(f"{distance:.2f}m")
        distance_label.setStyleSheet("color: #00ff00; font-size: 18px; font-family: 'Courier New'; font-weight: bold;")
        distance_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(distance_label)
        
        # Color indicator
        indicator = QFrame()
        indicator.setMinimumHeight(20)
        indicator.setMaximumHeight(20)
        indicator.setStyleSheet("border-radius: 4px;")
        layout.addWidget(indicator)
        
        frame.distance_label = distance_label
        frame.indicator = indicator
        frame.sensor_name = name
        
        return frame
    
    def get_color_for_distance(self, distance):
        """Get color based on distance reading."""
        if distance > 2.0:
            return "#00ff00"  # Green
        elif distance > 1.0:
            return "#ffff00"  # Yellow
        else:
            return "#ff0000"  # Red
    
    def update_sensors(self):
        """Update sensor readings with simulated data."""
        for name, data in self.sensor_data.items():
            # Simulate sensor drift
            data['distance'] += random.uniform(-0.3, 0.3)
            data['distance'] = max(0.2, min(5.0, data['distance']))
            
            widget = data['widget']
            color = self.get_color_for_distance(data['distance'])
            
            # Update distance label
            widget.distance_label.setText(f"{data['distance']:.2f}m")
            widget.distance_label.setStyleSheet(f"color: {color}; font-size: 18px; font-family: 'Courier New'; font-weight: bold;")
            
            # Update indicator
            widget.indicator.setStyleSheet(f"border-radius: 4px; background-color: {color};")


class RadarWidget(QFrame):
    """Widget for displaying object detection radar."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet("border: 2px solid #1a4d5c; border-radius: 8px; background-color: #0a1f2e;")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(300)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Object Detection Radar")
        title.setStyleSheet("color: #00d9ff; font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        # Radar canvas
        self.radar_display = RadarCanvas()
        layout.addWidget(self.radar_display)
    
    def paintEvent(self, event):
        """Override to ensure proper rendering."""
        super().paintEvent(event)


class RadarCanvas(QFrame):
    """Canvas for rendering the radar visualization."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(250)
        self.setStyleSheet("background-color: #051820; border-radius: 4px;")
        
        self.sweep_angle = 0
        self.detected_objects = []
        
        # Timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_radar)
        self.timer.start(50)  # Update every 50ms
    
    def update_radar(self):
        """Update radar display."""
        # Rotate sweep line
        self.sweep_angle = (self.sweep_angle + 3) % 360
        
        # Update object positions (simulated)
        if random.random() < 0.1:
            self.detected_objects = [
                {
                    'angle': random.uniform(0, 360),
                    'distance': random.uniform(0.2, 1.0),  # Normalized to 0-1
                    'threat': random.random() < 0.3,
                    'label': random.choice(['Boat', 'Obstacle', 'Vessel'])
                }
                for _ in range(random.randint(2, 5))
            ]
        
        self.update()
    
    def paintEvent(self, event):
        """Paint the radar visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) // 2 - 20
        
        # Draw background circles
        painter.setPen(QPen(QColor(26, 77, 92), 1, Qt.PenStyle.SolidLine))
        painter.setBrush(QColor(5, 24, 32))
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Draw range rings
        painter.setPen(QPen(QColor(26, 77, 92), 1, Qt.PenStyle.DashLine))
        for i in range(1, 4):
            ring_radius = radius * i // 3
            painter.drawEllipse(center_x - ring_radius, center_y - ring_radius,
                              ring_radius * 2, ring_radius * 2)
        
        # Draw crosshairs
        painter.setPen(QPen(QColor(26, 77, 92), 1))
        painter.drawLine(center_x - radius, center_y, center_x + radius, center_y)
        painter.drawLine(center_x, center_y - radius, center_x, center_y + radius)
        
        # Draw cardinal directions
        painter.setPen(QPen(QColor(0, 217, 255), 1))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(center_x - 10, center_y - radius - 15, "N")
        painter.drawText(center_x + radius + 10, center_y + 5, "E")
        painter.drawText(center_x - 10, center_y + radius + 20, "S")
        painter.drawText(center_x - radius - 25, center_y + 5, "W")
        
        # Draw ORACLE at center
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.drawEllipse(center_x - 5, center_y - 5, 10, 10)
        painter.setPen(QPen(QColor(0, 255, 0), 1))
        painter.drawLine(center_x, center_y, center_x, center_y - 15)
        
        # Draw sweep line
        sweep_rad = math.radians(self.sweep_angle)
        sweep_x = center_x + radius * math.cos(sweep_rad - math.pi / 2)
        sweep_y = center_y + radius * math.sin(sweep_rad - math.pi / 2)
        painter.setPen(QPen(QColor(0, 255, 100, 128), 2))
        painter.drawLine(center_x, center_y, int(sweep_x), int(sweep_y))
        
        # Draw detected objects
        for obj in self.detected_objects:
            angle_rad = math.radians(obj['angle'])
            distance_ratio = obj['distance']
            
            x = center_x + radius * distance_ratio * math.cos(angle_rad - math.pi / 2)
            y = center_y + radius * distance_ratio * math.sin(angle_rad - math.pi / 2)
            
            color = QColor(255, 0, 0) if obj['threat'] else QColor(0, 255, 0)
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            painter.drawEllipse(int(x) - 5, int(y) - 5, 10, 10)
            
            # Draw label
            painter.setPen(QPen(color, 1))
            painter.setFont(QFont("Arial", 8))
            painter.drawText(int(x) + 8, int(y) + 3, obj['label'])
        
        painter.end()


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ORACLE Autonomous Surveillance Dashboard")
        self.setGeometry(100, 100, 1400, 800)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #050f19;
            }
            QLabel {
                color: #e0e0e0;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Content layout
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(15)
        
        # Left side - Camera feed (75%)
        self.camera_widget = CameraFeedWidget()
        content_layout.addWidget(self.camera_widget, 3)
        
        # Right side (25%)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        
        # Ultrasonic widget
        self.ultrasonic_widget = UltrasonicWidget()
        right_layout.addWidget(self.ultrasonic_widget, 1)
        
        # Radar widget
        self.radar_widget = RadarWidget()
        right_layout.addWidget(self.radar_widget, 2)
        
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        content_layout.addWidget(right_widget, 1)
        
        main_layout.addLayout(content_layout, 1)
    
    def create_header(self):
        """Create the header bar."""
        header = QFrame()
        header.setStyleSheet("background-color: #0a1f2e; border-bottom: 2px solid #1a4d5c;")
        header.setMaximumHeight(60)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("ORACLE Autonomous Surveillance Dashboard")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #00d9ff;")
        layout.addWidget(title)
        
        # Status indicator
        status_frame = QFrame()
        status_frame.setMaximumWidth(150)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(10, 5, 10, 5)
        status_layout.setSpacing(8)
        
        status_indicator = QFrame()
        status_indicator.setMaximumSize(15, 15)
        status_indicator.setStyleSheet("background-color: #00ff00; border-radius: 7px;")
        status_layout.addWidget(status_indicator)
        
        status_text = QLabel("SYSTEM ACTIVE")
        status_text.setStyleSheet("color: #00ff00; font-weight: bold;")
        status_layout.addWidget(status_text)
        
        layout.addStretch()
        layout.addWidget(status_frame)
        
        return header


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
