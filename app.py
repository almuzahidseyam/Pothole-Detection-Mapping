import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import folium
from streamlit_folium import st_folium
import pandas as pd
import os
import plotly.express as px
import random
import time

st.set_page_config(page_title="Pothole Patrol Premium", page_icon="🛣️", layout="wide")

# --- Load YOLO Model ---
@st.cache_resource
def load_model():
    # Use standard YOLOv8 nano model as a lightweight fallback 
    # In production, replace 'yolov8n.pt' with a model trained specifically on pothole datasets (e.g. 'models/pothole_best.pt')
    model_path = 'models/best.pt' if os.path.exists('models/best.pt') else 'yolov8n.pt'
    return YOLO(model_path)

model = load_model()

# --- Sidebar Configuration ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3565/3565411.png", width=80)
st.sidebar.title("⚙️ System Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
st.sidebar.markdown("---")
st.sidebar.info("Developed for advanced urban infrastructure monitoring and geospatial analytics.")

# --- Main Dashboard ---
st.title("🛣️ Pothole Patrol: Enterprise Infrastructure Mapping")
st.markdown("Advanced AI pipeline for real-time road hazard detection, mapping, and analytics.")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📹 Video Analytics", "🗺️ Geospatial Map", "📊 Data Dashboard"])

# Initialize session state to hold cumulative data
if 'pothole_data' not in st.session_state:
    st.session_state['pothole_data'] = pd.DataFrame(columns=["Timestamp", "Latitude", "Longitude", "Confidence", "Severity"])

# --- TAB 1: Video Analytics ---
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("1. Upload Telemetry")
        uploaded_file = st.file_uploader("Upload Dashcam Feed (MP4/AVI)", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            st.video(uploaded_file)
            start_processing = st.button("🚀 Start AI Processing", type="primary", use_container_width=True)
        else:
            start_processing = False

    with col2:
        st.subheader("2. Live AI Vision Feed")
        stframe = st.empty()
        
        if start_processing and uploaded_file is not None:
            # Save uploaded video to a temp file for OpenCV
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            detected_list = []
            
            # Base GPS for simulation (Dhaka coordinates)
            base_lat, base_lon = 23.8103, 90.4125
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 5th frame to speed up the web demo
                if frame_count % 5 == 0:
                    # Run YOLOv8 inference
                    results = model(frame, conf=conf_threshold, verbose=False)
                    
                    # Draw bounding boxes on the frame
                    annotated_frame = results[0].plot()
                    
                    # Convert BGR (OpenCV) to RGB (Streamlit)
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(rgb_frame, channels="RGB", use_column_width=True)
                    
                    # Extract Detection Data
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        # Simulate GPS movement telemetry
                        lat = base_lat + random.uniform(-0.02, 0.02)
                        lon = base_lon + random.uniform(-0.02, 0.02)
                        severity = "High" if conf > 0.7 else ("Medium" if conf > 0.5 else "Low")
                        
                        detected_list.append({
                            "Timestamp": time.strftime("%H:%M:%S"),
                            "Latitude": lat,
                            "Longitude": lon,
                            "Confidence": round(conf, 2),
                            "Severity": severity
                        })
                        
                    # Update progress UI
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count} of {total_frames}...")
            
            cap.release()
            progress_bar.empty()
            status_text.success(f"✅ Processing Complete! Detected {len(detected_list)} potential hazards.")
            
            # Append new data to session state
            if detected_list:
                new_data = pd.DataFrame(detected_list)
                st.session_state['pothole_data'] = pd.concat([st.session_state['pothole_data'], new_data], ignore_index=True)

# --- TAB 2: Geospatial Map ---
with tab2:
    st.subheader("🗺️ Live Infrastructure Hazard Map")
    data = st.session_state['pothole_data']
    
    if not data.empty:
        # Center map on average coordinates
        avg_lat = data["Latitude"].mean()
        avg_lon = data["Longitude"].mean()
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles="CartoDB dark_matter")
        
        for idx, row in data.iterrows():
            color = "red" if row["Severity"] == "High" else ("orange" if row["Severity"] == "Medium" else "yellow")
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=7,
                popup=f"<b>Conf:</b> {row['Confidence']}<br><b>Severity:</b> {row['Severity']}",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8
            ).add_to(m)
        
        st_folium(m, width=1000, height=500)
        
        # Download GIS Data
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download GIS Data (CSV)", csv, "hazard_map.csv", "text/csv", type="primary")
    else:
        st.info("No geospatial data available. Please process a video in the 'Video Analytics' tab first.")

# --- TAB 3: Data Dashboard ---
with tab3:
    st.subheader("📊 Hazard Analytics Dashboard")
    data = st.session_state['pothole_data']
    
    if not data.empty:
        # KPI Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Hazards Detected", len(data))
        col2.metric("High Severity Hazards", len(data[data["Severity"] == "High"]))
        col3.metric("Avg Detection Confidence", f"{data['Confidence'].mean():.2f}")
        
        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Pie Chart
            fig_pie = px.pie(
                data, names="Severity", title="Hazard Severity Distribution", color="Severity", 
                color_discrete_map={"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#ffe066"}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with chart_col2:
            # Histogram
            fig_bar = px.histogram(
                data, x="Confidence", nbins=10, title="Confidence Score Distribution",
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("### Raw Detection Logs")
        st.dataframe(data.style.highlight_max(axis=0), use_container_width=True)
    else:
        st.info("No analytics available. Process a video to generate insights.")
