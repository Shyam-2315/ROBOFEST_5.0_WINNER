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
    QLabel, QFrame, QGridLayout, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QDialog
)
from PySide6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QFont, QImage, QBrush,
    QLinearGradient
)
from PySide6.QtCore import Qt, QTimer, QSize, QPoint, QRect, QThread, Signal
import numpy as np
import cv2
import os
import json
import psutil
import torch

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

    # Signal to broadcast FPS to other widgets (e.g. radar)
    fps_updated = Signal(float)
    # Signal emitted when an incident/alert occurs (for timeline/log)
    incident_generated = Signal(object)
    # Signal to indicate overall threat level: "normal", "warning", "critical"
    threat_level_changed = Signal(str)

    def __init__(self):
        super().__init__()
        # Base style; MainWindow will tint border based on threat level
        self.setStyleSheet("border: 2px solid #1a4d5c; border-radius: 8px; background-color: #0a1f2e;")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)
        
        # Threat summary strip (acts as status for current primary target)
        self.title_label = QLabel("Type of object: None   |   Threat: NORMAL")
        self.title_label.setStyleSheet(
            "color: #00d9ff; font-weight: bold; font-size: 14px; padding: 4px 0;"
        )
        layout.addWidget(self.title_label)
        
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
        # Front/primary ultrasonic (can be left or front as per wiring)
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

        # Recording state
        self.recording = False
        self.video_writer = None
        self.log_file = None
        self.record_start_time = None
        
        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS UI refresh

        # Recording controls
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 8, 0, 0)
        controls_layout.setSpacing(10)

        self.rec_button = QPushButton("REC")
        self.rec_button.setStyleSheet("background-color: #550000; color: #ffffff; font-weight: bold;")
        self.rec_button.clicked.connect(self.start_recording)
        controls_layout.addWidget(self.rec_button)

        self.stop_button = QPushButton("STOP")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_recording)
        controls_layout.addWidget(self.stop_button)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

    def start_recording(self):
        """Start video + detection logging to disk."""
        if self.recording:
            return

        # Ensure we have a valid frame size
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or config.FRAME_WIDTH)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or config.FRAME_HEIGHT)
        fps = 20.0

        os.makedirs("recordings", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join("recordings", f"oracle_{timestamp}.mp4")
        log_path = os.path.join("recordings", f"oracle_{timestamp}.jsonl")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.record_start_time = datetime.now()
        self.recording = True

        self.rec_button.setStyleSheet("background-color: #ff0000; color: #ffffff; font-weight: bold;")
        self.stop_button.setEnabled(True)

    def stop_recording(self):
        """Stop recording and close resources."""
        if not self.recording:
            return

        self.recording = False

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

        self.rec_button.setStyleSheet("background-color: #550000; color: #ffffff; font-weight: bold;")
        self.stop_button.setEnabled(False)
    
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
        raw_distance = self.ultrasonic.get_distance()
        if raw_distance is None:
            distance_value = 9999.0
            distance_display = "N/A"
        else:
            distance_value = float(raw_distance)
            distance_display = int(distance_value)
        fps = self.fps_counter.update()

        alert_text = None
        target_center = None
        current_label = "None"

        if target:
            x1, y1, x2, y2 = target["bbox"]
            label = target["label"]
            current_label = label

            servo_x, servo_y, cx, cy = self.target_lock.compute((x1, y1, x2, y2))
            self.servo.move(servo_x, servo_y)

            target_center = (cx, cy)
            alert_text = self.alert_system.check_alert(label, cx, cy, distance_value)

            # Emit incident event for the log/timeline
            if alert_text:
                incident = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "label": label,
                    "distance_cm": float(distance_value),
                    "alert": alert_text,
                }
                self.incident_generated.emit(incident)

        # Update title based on current target and alert
        if current_label != "None":
            threat_level = "CRITICAL" if alert_text else "WARNING"
            self.title_label.setText(f"Type of object: {current_label}   |   Threat: {threat_level}")
        else:
            threat_level = "NORMAL"
            self.title_label.setText("Type of object: None   |   Threat: NORMAL")
        
        # Notify rest of UI about current threat level
        self.threat_level_changed.emit(threat_level.lower())

        # Emit FPS to any listeners (for GPU/System FPS display)
        self.fps_updated.emit(float(fps))

        # If recording, write video frame and metadata
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)
            if self.log_file is not None:
                record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "fps": float(fps),
                    "distance_cm": float(distance_value),
                    "motion": bool(motion_flag),
                    "alert": alert_text,
                    "detections": detections,
                }
                self.log_file.write(json.dumps(record) + "\n")
        
        # Use shared OpenCV dashboard renderer (same as main.py)
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
        title = QLabel("Ultrasonic Distance Readings (10s average)")
        title.setStyleSheet("color: #00d9ff; font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        # Real ultrasonic sensors: left and right
        self.left_sensor = UltrasonicSensor(port="/dev/ttyTHS1")
        self.right_sensor = UltrasonicSensor(port="/dev/ttyTHS2")

        # Data accumulators for 10s averages
        self.left_samples: list[float] = []
        self.right_samples: list[float] = []
        self.tick_count = 0

        self.sensor_frames = {}
        
        grid = QGridLayout()
        grid.setSpacing(10)
        
        for idx, name in enumerate(["Left Ultrasonic", "Right Ultrasonic"]):
            sensor_frame = self.create_sensor_display(name)
            self.sensor_frames[name] = sensor_frame
            grid.addWidget(sensor_frame, idx, 0)
        
        layout.addLayout(grid)
        layout.addStretch()
        
        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_sensors)
        self.timer.start(1000)  # Update every second
    
    def create_sensor_display(self, name):
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
        distance_label = QLabel("N/A")
        distance_label.setStyleSheet("color: #e0e0e0; font-size: 18px; font-family: 'Courier New'; font-weight: bold;")
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
    
    def get_color_for_distance(self, distance_cm: float):
        """Get color based on distance reading."""
        # thresholds in cm
        if distance_cm > 200:
            return "#00ff00"  # Green
        elif distance_cm > 100:
            return "#ffff00"  # Yellow
        else:
            return "#ff0000"  # Red
    
    def update_sensors(self):
        """Update sensor readings and show 10s averages."""
        self.tick_count += 1

        # Helper to process one side
        def process_side(name: str, sensor: UltrasonicSensor, samples: list[float]):
            widget = self.sensor_frames[name]

            if not sensor.connected:
                widget.distance_label.setText("No ultrasonic connected")
                widget.distance_label.setStyleSheet(
                    "color: #888888; font-size: 14px; font-family: 'Courier New'; font-weight: bold;"
                )
                widget.indicator.setStyleSheet("border-radius: 4px; background-color: #444444;")
                samples.clear()
                return

            d = sensor.get_distance()
            if d is None:
                # No valid reading yet
                return

            samples.append(float(d))

            # Every 10 seconds (10 ticks at 1s interval) compute and display average
            if self.tick_count % 10 == 0 and samples:
                avg = sum(samples) / len(samples)
                color = self.get_color_for_distance(avg)
                widget.distance_label.setText(f"{avg:.1f} cm")
                widget.distance_label.setStyleSheet(
                    f"color: {color}; font-size: 18px; font-family: 'Courier New'; font-weight: bold;"
                )
                widget.indicator.setStyleSheet(f"border-radius: 4px; background-color: {color};")
                samples.clear()

        process_side("Left Ultrasonic", self.left_sensor, self.left_samples)
        process_side("Right Ultrasonic", self.right_sensor, self.right_samples)


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
        
        # Title (shows GPU/system FPS)
        self.title_label = QLabel("GPU / System FPS: --.-")
        self.title_label.setStyleSheet("color: #00d9ff; font-weight: bold; font-size: 12px;")
        layout.addWidget(self.title_label)
        
        # Radar canvas
        self.radar_display = RadarCanvas()
        layout.addWidget(self.radar_display)

    def set_fps(self, fps: float) -> None:
        """Update the displayed FPS value."""
        self.title_label.setText(f"GPU / System FPS: {fps:.1f}")
    
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
                    "angle": random.uniform(0, 360),
                    "distance": random.uniform(0.2, 1.0),  # Normalized to 0-1
                    "threat": random.random() < 0.3,
                    "label": random.choice(["Boat", "Obstacle", "Vessel"]),
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
            painter.drawEllipse(
                center_x - ring_radius,
                center_y - ring_radius,
                ring_radius * 2,
                ring_radius * 2,
            )
        
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
            angle_rad = math.radians(obj["angle"])
            distance_ratio = obj["distance"]
            
            x = center_x + radius * distance_ratio * math.cos(angle_rad - math.pi / 2)
            y = center_y + radius * distance_ratio * math.sin(angle_rad - math.pi / 2)
            
            color = QColor(255, 0, 0) if obj["threat"] else QColor(0, 255, 0)
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            painter.drawEllipse(int(x) - 5, int(y) - 5, 10, 10)
            
            # Draw label
            painter.setPen(QPen(color, 1))
            painter.setFont(QFont("Arial", 8))
            painter.drawText(int(x) + 8, int(y) + 3, obj["label"])
        
        painter.end()


class SystemHealthWidget(QFrame):
    """Widget to display basic system health (CPU, RAM, GPU)."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet("border: 2px solid #1a4d5c; border-radius: 8px; background-color: #0a1f2e;")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(8)

        title = QLabel("System Health")
        title.setStyleSheet("color: #00d9ff; font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        self.cpu_label = QLabel("CPU: -- %")
        self.ram_label = QLabel("RAM: -- %")
        self.gpu_label = QLabel("GPU: -- % (VRAM -- / -- GB)")

        for lbl in (self.cpu_label, self.ram_label, self.gpu_label):
            lbl.setStyleSheet("color: #e0e0e0; font-size: 11px;")
            layout.addWidget(lbl)

        layout.addStretch()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)

    def update_stats(self):
        # CPU and RAM via psutil
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent

        cpu_color = "#ff0000" if cpu > 90 else "#00ff00"
        ram_color = "#ff0000" if ram > 90 else "#00ff00"

        self.cpu_label.setText(f"CPU: {cpu:.0f} %")
        self.cpu_label.setStyleSheet(f"color: {cpu_color}; font-size: 11px;")

        self.ram_label.setText(f"RAM: {ram:.0f} %")
        self.ram_label.setStyleSheet(f"color: {ram_color}; font-size: 11px;")

        # Basic GPU info via torch (if available)
        if torch.cuda.is_available():
            try:
                gpu_idx = 0
                total = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 3)
                used = torch.cuda.memory_allocated(gpu_idx) / (1024 ** 3)
                util = (used / total) * 100 if total > 0 else 0
                gpu_color = "#ff0000" if util > 90 else "#00ff00"
                self.gpu_label.setText(f"GPU: {util:.0f} % (VRAM {used:.1f} / {total:.1f} GB)")
                self.gpu_label.setStyleSheet(f"color: {gpu_color}; font-size: 11px;")
            except Exception:
                self.gpu_label.setText("GPU: N/A")
                self.gpu_label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        else:
            self.gpu_label.setText("GPU: Not available")
            self.gpu_label.setStyleSheet("color: #e0e0e0; font-size: 11px;")


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
        
        # Radar widget (also shows GPU/System FPS)
        self.radar_widget = RadarWidget()
        right_layout.addWidget(self.radar_widget, 2)

        # System health monitor (CPU/GPU/RAM)
        self.health_widget = SystemHealthWidget()
        right_layout.addWidget(self.health_widget, 1)

        # Threat timeline / incident log
        self.incident_list = QListWidget()
        self.incident_list.setStyleSheet(
            "background-color: #051820; color: #e0e0e0; border: 1px solid #1a4d5c;"
        )
        self.incident_list.itemClicked.connect(self.show_incident_snapshot)
        right_layout.addWidget(self.incident_list, 2)
        
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        content_layout.addWidget(right_widget, 1)
        
        main_layout.addLayout(content_layout, 1)

        # Connect camera FPS updates to radar display
        self.camera_widget.fps_updated.connect(self.radar_widget.set_fps)
        # Connect incidents from camera to incident log
        self.camera_widget.incident_generated.connect(self.add_incident)
        # React to threat level changes to tint UI
        self.camera_widget.threat_level_changed.connect(self.update_threat_state)

    def add_incident(self, incident: object) -> None:
        """Add an incident entry to the timeline list."""
        if not isinstance(incident, dict):
            return
        ts = incident.get("timestamp", "")
        label = incident.get("label", "")
        dist = incident.get("distance_cm", 0.0)
        alert = incident.get("alert", "")
        text = f"[{ts}] {alert} - {label} @ {dist:.1f} cm"
        item = QListWidgetItem(text)
        # Store raw data for later use (e.g., snapshot, export)
        item.setData(Qt.ItemDataRole.UserRole, incident)
        self.incident_list.addItem(item)
        self.incident_list.scrollToBottom()

    def show_incident_snapshot(self, item: QListWidgetItem) -> None:
        """Display a simple dialog with incident details (placeholder for snapshot)."""
        data = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Incident details")
        layout = QVBoxLayout(dlg)
        label = QLabel(
            f"Time: {data.get('timestamp','')}\n"
            f"Object: {data.get('label','')}\n"
            f"Alert: {data.get('alert','')}\n"
            f"Distance: {data.get('distance_cm',0.0):.1f} cm"
        )
        label.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(label)
        dlg.setLayout(layout)
        dlg.exec()
    
    def update_threat_state(self, level: str) -> None:
        """Update global UI colors based on current threat level."""
        level = (level or "normal").lower()

        if level == "critical":
            header_color = "#3c0000"
            status_color = "#ff0000"
            status_text = "SYSTEM CRITICAL"
            cam_border = "#ff0000"
        elif level == "warning":
            header_color = "#3c2a00"
            status_color = "#ffcc00"
            status_text = "SYSTEM WARNING"
            cam_border = "#ffcc00"
        else:
            header_color = "#0a1f2e"
            status_color = "#00ff00"
            status_text = "SYSTEM NORMAL"
            cam_border = "#1a4d5c"

        # Header background
        self.header_frame.setStyleSheet(
            f"background-color: {header_color}; border-bottom: 2px solid #1a4d5c;"
        )
        # Status light + text
        self.status_indicator.setStyleSheet(
            f"background-color: {status_color}; border-radius: 7px;"
        )
        self.status_text.setText(status_text)
        self.status_text.setStyleSheet(f"color: {status_color}; font-weight: bold;")

        # Camera widget border tint
        self.camera_widget.setStyleSheet(
            f"border: 2px solid {cam_border}; border-radius: 8px; background-color: #0a1f2e;"
        )
    
    def create_header(self):
        """Create the header bar."""
        header = QFrame()
        self.header_frame = header
        header.setStyleSheet("background-color: #0a1f2e; border-bottom: 2px solid #1a4d5c;")
        header.setMaximumHeight(70)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 8, 20, 8)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("ORACLE Naval Surveillance & Defense Console")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #00d9ff;")
        layout.addWidget(title)

        # Mode selector
        mode_frame = QFrame()
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(6)

        self.mode_buttons = []
        for mode_name in ["Harbor Patrol", "Search & Rescue", "Defense"]:
            btn = QPushButton(mode_name)
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton { color: #e0e0e0; background-color: #12293a; border-radius: 4px; padding: 4px 8px; }"
                "QPushButton:checked { background-color: #00d9ff; color: #000000; font-weight: bold; }"
            )
            btn.clicked.connect(self.on_mode_clicked)
            self.mode_buttons.append(btn)
            mode_layout.addWidget(btn)

        # Default mode
        if self.mode_buttons:
            self.mode_buttons[0].setChecked(True)

        layout.addWidget(mode_frame)
        
        # Status indicator
        status_frame = QFrame()
        status_frame.setMaximumWidth(190)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(10, 5, 10, 5)
        status_layout.setSpacing(8)
        
        status_indicator = QFrame()
        status_indicator.setMaximumSize(15, 15)
        status_indicator.setStyleSheet("background-color: #00ff00; border-radius: 7px;")
        status_layout.addWidget(status_indicator)
        self.status_indicator = status_indicator
        
        status_text = QLabel("SYSTEM NORMAL")
        status_text.setStyleSheet("color: #00ff00; font-weight: bold;")
        status_layout.addWidget(status_text)
        self.status_text = status_text
        
        layout.addStretch()
        layout.addWidget(status_frame)
        
        return header

    def on_mode_clicked(self):
        """Ensure only one mode button is active at a time."""
        sender = self.sender()
        for btn in self.mode_buttons:
            if btn is not sender:
                btn.setChecked(False)


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
