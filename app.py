import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import folium
from streamlit_folium import st_folium
import pandas as pd
import os
import plotly.express as px
import time
import math

st.set_page_config(page_title="Pothole Patrol Enterprise", page_icon="🛣️", layout="wide")

# Custom CSS for Premium UI
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #FF4B4B;}
    .stButton>button {border-radius: 20px; font-weight: bold; padding: 0.5rem 2rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; font-size: 1.2rem;}
    </style>
""", unsafe_allow_html=True)

# --- Load YOLO Model ---
@st.cache_resource
def load_model():
    model_path = 'models/best.pt' if os.path.exists('models/best.pt') else 'models/yolov8n.pt'
    return YOLO(model_path)

model = load_model()

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3565/3565411.png", width=80)
st.sidebar.title("⚙️ Engine Settings")
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
processing_speed = st.sidebar.radio("Processing Speed", ["Fast (Every 5th Frame)", "Max Accuracy (Every Frame)"])
frame_skip = 5 if processing_speed.startswith("Fast") else 1

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Pothole Patrol Enterprise. All rights reserved.")

# --- Main ---
st.title("🛣️ Pothole Patrol: AI Infrastructure Intelligence")
st.markdown("Enterprise-grade pipeline for real-time hazard detection and GIS analytics.")

tab1, tab2, tab3 = st.tabs(["📹 AI Video Vision", "🗺️ GIS Hazard Map", "📊 Analytics Dashboard"])

if 'pothole_data' not in st.session_state:
    st.session_state['pothole_data'] = pd.DataFrame(columns=["Timestamp", "Latitude", "Longitude", "Confidence", "Severity"])

with tab1:
    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.subheader("1. Data Ingestion")
        uploaded_file = st.file_uploader("Upload Dashcam Feed (MP4/AVI)", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            st.video(uploaded_file)
            start_processing = st.button("🚀 Initialize AI Engine", type="primary", use_container_width=True)
        else:
            start_processing = False

    with col2:
        st.subheader("2. Real-Time Vision Feed")
        stframe = st.empty()
        
        if start_processing and uploaded_file is not None:
            # FIX: Properly handle temp file to avoid Windows PermissionErrors
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                temp_filename = tfile.name
            
            cap = cv2.VideoCapture(temp_filename)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            detected_list = []
            
            # Base GPS for simulation (Dhaka coordinates moving slightly)
            current_lat, current_lon = 23.8103, 90.4125
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % frame_skip == 0:
                    results = model(frame, conf=conf_threshold, verbose=False)
                    annotated_frame = results[0].plot(line_width=2)
                    
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(rgb_frame, channels="RGB", use_container_width=True)
                    
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        # Simulate car moving slowly
                        current_lat += 0.0001
                        current_lon += 0.00005
                        severity = "Critical" if conf > 0.75 else ("Warning" if conf > 0.5 else "Minor")
                        
                        detected_list.append({
                            "Timestamp": time.strftime("%H:%M:%S"),
                            "Latitude": current_lat,
                            "Longitude": current_lon,
                            "Confidence": round(conf, 2),
                            "Severity": severity
                        })
                        
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    status_text.text(f"Analyzing Telemetry: Frame {frame_count}/{total_frames}...")
            
            cap.release()
            
            # Cleanup temp file gracefully
            try:
                os.remove(temp_filename)
            except OSError:
                pass
                
            progress_bar.empty()
            status_text.success(f"✅ Telemetry Analysis Complete! Identified {len(detected_list)} structural anomalies.")
            
            if detected_list:
                new_data = pd.DataFrame(detected_list)
                st.session_state['pothole_data'] = pd.concat([st.session_state['pothole_data'], new_data], ignore_index=True)

with tab2:
    st.subheader("🗺️ Live GIS Infrastructure Map")
    data = st.session_state['pothole_data']
    
    if not data.empty:
        avg_lat = data["Latitude"].mean()
        avg_lon = data["Longitude"].mean()
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=15, tiles="CartoDB dark_matter")
        
        for idx, row in data.iterrows():
            color = "#ff4b4b" if row["Severity"] == "Critical" else ("#ffa500" if row["Severity"] == "Warning" else "#ffe066")
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=8,
                popup=f"<b>Confidence:</b> {row['Confidence']}<br><b>Severity:</b> {row['Severity']}",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9
            ).add_to(m)
        
        st_folium(m, width=1200, height=600)
        
        st.download_button("📥 Export GIS Coordinates (CSV)", data.to_csv(index=False).encode('utf-8'), "gis_hazard_data.csv", "text/csv", type="primary")
    else:
        st.info("No geospatial data available. Initialize AI engine in the Vision tab.")

with tab3:
    st.subheader("📊 Executive Analytics Dashboard")
    data = st.session_state['pothole_data']
    
    if not data.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Anomalies", len(data))
        col2.metric("Critical Hazards", len(data[data["Severity"] == "Critical"]))
        col3.metric("Warning Hazards", len(data[data["Severity"] == "Warning"]))
        col4.metric("AI Mean Confidence", f"{data['Confidence'].mean():.2f}")
        
        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            fig_pie = px.pie(
                data, names="Severity", title="Structural Severity Breakdown", hole=0.4,
                color="Severity", color_discrete_map={"Critical": "#ff4b4b", "Warning": "#ffa500", "Minor": "#ffe066"}
            )
            fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with chart_col2:
            fig_bar = px.histogram(
                data, x="Confidence", nbins=15, title="AI Confidence Distribution",
                color_discrete_sequence=['#ff4b4b']
            )
            fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("### 📋 Detailed Inspection Logs")
        st.dataframe(data.style.highlight_max(axis=0), use_container_width=True)
    else:
        st.info("Analytics engine is awaiting data.")
