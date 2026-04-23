# 🚁 ORACLE: Autonomous Surveillance & Defense System

### 🏆 Robofest 5.0 – Winning Project

---

## 📌 Overview

**ORACLE** is an advanced AI-powered surveillance and defense system designed for real-time monitoring, threat detection, and autonomous response in dynamic environments.

Built for **Robofest 5.0**, this system integrates **Computer Vision, Embedded Systems, and AI** to deliver a fully functional **Naval Surveillance Dashboard** running on NVIDIA Jetson.

---

## 🎯 Key Features

* 🎥 **Real-Time Camera Feed**

  * USB camera integration on Jetson
  * Live video streaming in PySide6 dashboard

* 🧠 **AI Object Detection (YOLOv8)**

  * Detects:

    * 👤 Humans
    * 🚤 Boats / Ships
    * 🚧 Obstacles / Debris
  * Optimized for real-time inference

* 🎯 **Target Tracking & Lock System**

  * Intelligent object selection
  * Servo-based tracking using bounding box center

* 📡 **Ultrasonic Sensor Integration**

  * Distance measurement (left & right sensors)
  * Real-time obstacle awareness

* 📊 **Advanced Dashboard UI**

  * Built with PySide6
  * Radar visualization
  * Threat level indication (Normal / Warning / Critical)

* ⚠️ **Alert & Incident System**

  * Automatic threat classification
  * Timeline logging of events

* 💾 **Recording System**

  * Save video feed
  * Store detection logs (JSON format)

* 🖥️ **System Health Monitoring**

  * CPU, RAM, GPU usage
  * FPS tracking

---

## 🧱 Tech Stack

| Category         | Technology                    |
| ---------------- | ----------------------------- |
| AI / ML          | YOLOv8 (Ultralytics), PyTorch |
| Backend          | Python                        |
| UI               | PySide6 (Qt)                  |
| Computer Vision  | OpenCV                        |
| Embedded         | NVIDIA Jetson                 |
| Sensors          | Ultrasonic (Serial)           |
| Hardware Control | Servo Motors                  |
| Utilities        | psutil                        |

---

## 🏗️ Project Architecture

```bash
project/
│
├── main.py                  # Main application entry
├── config.py               # Configuration settings
│
├── core/
│   ├── vision.py           # YOLO detection system
│   ├── motion_detector.py
│   ├── object_tracker.py
│   ├── target_lock.py
│
├── sensors/
│   └── ultrasonic.py       # Ultrasonic sensor logic
│
├── hardware/
│   └── servo_controller.py # Servo control
│
├── utils/
│   ├── fps_counter.py
│   ├── alert_system.py
│
├── ui/
│   └── dashboard.py        # OpenCV overlay rendering
│
└── recordings/             # Saved videos & logs
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/oracle-surveillance.git
cd oracle-surveillance
```

### 2️⃣ Install Dependencies

```bash
sudo apt update
sudo apt install python3-opencv python3-pip -y

pip3 install ultralytics torch torchvision
pip3 install PySide6 psutil
```

---

## 📷 Camera Setup (Jetson)

Ensure camera is detected:

```bash
ls /dev/video*
```

Use optimized pipeline in code:

```python
cv2.VideoCapture(0, cv2.CAP_V4L2)
```

---

## 🚀 Running the Project

```bash
python3 main.py
```

---

## 🧠 AI Detection Classes

Currently detecting:

* person
* boat
* car / truck (proxy for vessels & obstacles)

---

## ⚡ Performance Optimizations

* YOLOv8 Nano (`yolov8n.pt`) for fast inference
* Frame skipping (`DETECTION_INTERVAL = 3`)
* Resolution: `640x480`
* GStreamer pipeline support for Jetson

---

## 📸 System Output

* Live annotated video feed
* Radar-based object visualization
* Threat alerts with timestamps
* Sensor-based distance readings

---

## 🔥 Future Improvements

* 🎯 Custom-trained maritime detection model
* ⚡ TensorRT optimization for Jetson
* 🚁 Pixhawk autonomous navigation integration
* 📡 Multi-camera fusion
* 🌐 Remote monitoring dashboard

---

## 🏆 Achievement

🥇 **Winner – Robofest 5.0**
Recognized for innovation in **AI-based autonomous surveillance systems**.

---

## 👨‍💻 Team

* Shyam Patel (Software Guy)
* Sneh Moradiya (Electrical Guy)
* Yasmin Sekh (Documentation)
* Sneh Brahmbhatt (Mechnical Guy)

---

## 📜 License

This project is licensed under the MIT License.

---

## 💡 Inspiration

Designed to push the boundaries of **AI + Robotics + Defense Systems** for real-world applications.

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!

---
