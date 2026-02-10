# 🚗 Pothole Patrol: AI-Powered Detection & Web Mapping

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer%20Vision-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![License](https://img.shields.io/badge/License-MIT-green)

> An automated computer vision pipeline that detects potholes from dashcam footage using YOLO and dynamically maps their coordinates on an interactive web map.

<div align="center">
  <!-- TODO: Add a GIF of your working project here -->
  <img src="https://via.placeholder.com/800x400?text=[Insert+Your+Demo+GIF/Video+Here]" alt="Project Demo GIF">
</div>

## 📌 Problem Statement
Potholes cause severe vehicle damage and road accidents globally. Identifying and mapping them manually is slow and inefficient. This project automates the process by analyzing dashcam videos, detecting potholes in real-time, and plotting their locations on a live map for urban planning and public safety.

## ✨ Key Features
- **Real-Time Detection:** Utilizes YOLOv8 for fast and highly accurate pothole detection.
- **Dynamic Web Mapping:** Automatically extracts location data and plots red hazard pins on an interactive map (Folium/Google Maps).
- **Interactive Dashboard:** A clean, user-friendly web interface built with Streamlit to upload videos and view results instantly.
- **Exportable Data:** Download the detected pothole coordinates as a CSV file for further analysis.

## 🧠 System Architecture

```mermaid
flowchart LR
    A[Input Video/Image] --> B(YOLOv8 Object Detection)
    B --> C{Pothole Detected?}
    C -- Yes --> D[Extract Frame & Coordinates]
    C -- No --> E[Skip Frame]
    D --> F[Streamlit Web App]
    D --> G[Plot on Interactive Map]
```

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/almuzahidseyam/Pothole-Detection-Mapping.git
   cd Pothole-Detection-Mapping
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## 🛠️ Tech Stack
- **Deep Learning:** Ultralytics YOLOv8
- **Computer Vision:** OpenCV
- **Frontend/Dashboard:** Streamlit
- **Mapping:** Folium / Streamlit-Folium

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License
This project is licensed under the MIT License.
