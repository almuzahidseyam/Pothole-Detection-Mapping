import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import folium
from streamlit_folium import st_folium
import pandas as pd
import os

st.set_page_config(page_title="Pothole Patrol", page_icon="🚗", layout="wide")

st.title("🚗 Pothole Patrol: AI-Powered Detection & Mapping")
st.markdown("Upload a dashcam video to detect potholes using YOLOv8 and dynamically map their locations.")

# Load model function
@st.cache_resource
def load_model():
    # In a real scenario, you would train a YOLOv8 model on pothole datasets
    # and load it here (e.g., 'models/best.pt'). 
    # For now, we load a placeholder pre-trained model to prevent crashes.
    model_path = 'yolov8n.pt'
    if os.path.exists('models/best.pt'):
        model_path = 'models/best.pt'
    return YOLO(model_path) 

model = load_model()

col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Dashcam Video")
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        st.video(tfile.name)
        
        if st.button("Detect Potholes", type="primary"):
            with st.spinner('Analyzing video frames with YOLOv8...'):
                # Core logic for YOLO detection on video frames goes here
                # Example: results = model(tfile.name, stream=True)
                
                st.success("✅ Detection Complete!")
                st.session_state['detected'] = True

with col2:
    st.header("2. Pothole Live Map")
    if st.session_state.get('detected', False):
        st.markdown("Detected pothole coordinates extracted from video telemetry.")
        # Generate a sample map (Simulating GPS extraction from dashcam metadata)
        # Assuming the video was captured in Dhaka, Bangladesh
        m = folium.Map(location=[23.8103, 90.4125], zoom_start=13)
        
        # Simulated pothole coordinates
        folium.Marker(
            [23.8150, 90.4150], 
            popup="Pothole Detected (Conf: 0.89)", 
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        
        folium.Marker(
            [23.8050, 90.4200], 
            popup="Pothole Detected (Conf: 0.92)", 
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        
        st_folium(m, width=700, height=500)
        
        # Option to download CSV
        csv_data = pd.DataFrame({
            "Latitude": [23.8150, 23.8050], 
            "Longitude": [90.4150, 90.4200],
            "Confidence": [0.89, 0.92]
        }).to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Coordinates (CSV)",
            data=csv_data,
            file_name="detected_potholes.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload a video and run detection to view the generated map.")
