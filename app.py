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

st.set_page_config(page_title="Pothole Patrol Enterprise", page_icon="🛣️", layout="wide")

# --- Custom CSS for Enterprise UI ---
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #FF4B4B;}
    .stButton>button {border-radius: 8px; font-weight: bold; border: 1px solid #FF4B4B;}
    .stButton>button:hover {background-color: #FF4B4B; color: white;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; font-size: 1.2rem;}
    </style>
""", unsafe_allow_html=True)

# --- Robust State Management ---
if 'pothole_data' not in st.session_state:
    st.session_state['pothole_data'] = pd.DataFrame()
if 'processed_video_path' not in st.session_state:
    st.session_state['processed_video_path'] = None
if 'current_file_name' not in st.session_state:
    st.session_state['current_file_name'] = ""

# --- Load YOLO Model Securely ---
@st.cache_resource(show_spinner="Loading Deep Learning Weights...")
def load_model():
    model_path = 'models/best.pt' if os.path.exists('models/best.pt') else 'models/yolov8n.pt'
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Critical Error: Unable to load AI model. Details: {e}")
    st.stop()

# --- Sidebar Configuration ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3565/3565411.png", width=80)
st.sidebar.title("⚙️ AI Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
frame_skip = st.sidebar.select_slider(
    "Processing Engine Speed", 
    options=[1, 2, 5, 10], 
    value=2, 
    help="1 = Max Accuracy (processes every frame), 10 = Max Speed (processes every 10th frame)"
)
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Pothole Patrol Analytics. Secure Build.")

# --- Main Dashboard ---
st.title("🛣️ Pothole Patrol: AI Infrastructure Intelligence")
st.markdown("Enterprise-grade pipeline for real-time hazard detection, video rendering, and GIS analytics.")

tab1, tab2, tab3 = st.tabs(["📹 AI Vision Processing", "🗺️ GIS Hazard Map", "📊 Analytics Dashboard"])

# --- TAB 1: Video Analytics with Video Export ---
with tab1:
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.subheader("1. Data Ingestion")
        uploaded_file = st.file_uploader("Upload Dashcam Feed (MP4/AVI)", type=["mp4", "avi", "mov"])
        
        # Flaw Fix: Reset memory state automatically if a NEW file is uploaded
        if uploaded_file is not None:
            if uploaded_file.name != st.session_state['current_file_name']:
                st.session_state['current_file_name'] = uploaded_file.name
                st.session_state['pothole_data'] = pd.DataFrame()
                st.session_state['processed_video_path'] = None
            
            st.video(uploaded_file)
            start_processing = st.button("🚀 Initialize AI Engine", type="primary", use_container_width=True)
        else:
            start_processing = False

    with col2:
        st.subheader("2. Real-Time Vision Feed")
        stframe = st.empty()
        
        if start_processing and uploaded_file is not None:
            temp_input = None
            temp_output = None
            try:
                # Bug Fix: Handle temp files safely with Try/Finally
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile_in:
                    tfile_in.write(uploaded_file.read())
                    temp_input = tfile_in.name
                
                # Setup output video writer for exporting the processed video
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                cap = cv2.VideoCapture(temp_input)
                
                if not cap.isOpened():
                    st.error("Engine Error: Video feed could not be initialized.")
                else:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    frame_count = 0
                    detected_list = []
                    
                    # Simulated GPS starting point
                    current_lat, current_lon = 23.8103, 90.4125
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        
                        # Process target frames based on speed setting
                        if frame_count % frame_skip == 0:
                            results = model(frame, conf=conf_threshold, verbose=False)
                            annotated_frame = results[0].plot(line_width=2)
                            
                            # Write annotated frame to the output video
                            out_writer.write(annotated_frame)
                            
                            # Display on Web UI
                            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            stframe.image(rgb_frame, channels="RGB", use_container_width=True)
                            
                            # Log Geospatial Data
                            for box in results[0].boxes:
                                conf = float(box.conf[0])
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
                        else:
                            # Keep non-processed frames in output for smooth playback
                            out_writer.write(frame)
                    
                    # Release resources safely
                    cap.release()
                    out_writer.release()
                    
                    # Store output path in session
                    st.session_state['processed_video_path'] = temp_output
                    progress_bar.empty()
                    status_text.success(f"✅ Telemetry Analysis Complete! Identified {len(detected_list)} structural anomalies.")
                    
                    if detected_list:
                        new_data = pd.DataFrame(detected_list)
                        st.session_state['pothole_data'] = new_data
            
            except Exception as e:
                st.error(f"⚠️ Critical Engine Failure: {str(e)}")
            finally:
                # Bug Fix: Ensure memory cleanup even if processing fails
                if temp_input and os.path.exists(temp_input):
                    try:
                        os.remove(temp_input)
                    except OSError:
                        pass
        
        # Download Rendered Video Button
        if st.session_state.get('processed_video_path') and os.path.exists(st.session_state['processed_video_path']):
            with open(st.session_state['processed_video_path'], 'rb') as f:
                st.download_button(
                    label="📥 Download Rendered AI Video (MP4)", 
                    data=f.read(), 
                    file_name="ai_rendered_telemetry.mp4", 
                    mime="video/mp4", 
                    type="primary"
                )

# --- TAB 2: Geospatial Map ---
with tab2:
    st.subheader("🗺️ Live GIS Infrastructure Map")
    data = st.session_state.get('pothole_data', pd.DataFrame())
    
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
        st.download_button("📥 Export GIS Coordinates (CSV)", data.to_csv(index=False).encode('utf-8'), "gis_hazard_data.csv", "text/csv")
    else:
        st.info("No geospatial data available. Initialize AI engine in the Vision tab.")

# --- TAB 3: Analytics Dashboard ---
with tab3:
    st.subheader("📊 Executive Analytics Dashboard")
    data = st.session_state.get('pothole_data', pd.DataFrame())
    
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
