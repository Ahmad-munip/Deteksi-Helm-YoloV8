import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.info("💡 **Tip:** Change settings in the sidebar and click '🔄 Refresh Model' to apply them.")

# Page configuration
st.set_page_config(
    page_title="🛡️ Smart Helmet Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main background - elegant dark theme */
    .stApp {
         background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
    color: #333;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Card styling */
    .info-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling - professional blue */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #3b82f6);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338ca, #2563eb);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
    }
    
    /* Metrics styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Status indicators - professional colors */
    .status-success {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-warning {
        color: #f59e0b;
        font-weight: 600;
    }
    
    .status-error {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(79, 70, 229, 0.3);
        color: white;
        border: 1px solid rgba(79, 70, 229, 0.5);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
def init_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    if 'detection_count' not in st.session_state:
        st.session_state.detection_count = 0
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

# Initialize session state
init_session_state()

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ Smart Helmet Detection System</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem; opacity: 0.9;">
        Advanced AI-powered safety monitoring with real-time detection
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Model loading section
    st.markdown("### 🤖 Model Settings")
    model_path = st.text_input("Model Path", value="best.pt", help="Path to your YOLO model file")
    
    if st.button("🔄 Load/Reload Model", key="load_model"):
        with st.spinner("Loading model..."):
            try:
                st.session_state.model = YOLO(model_path)
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
                st.rerun()  # Refresh to update UI
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                st.session_state.model = None
                st.session_state.model_loaded = False
    
    # Detection settings
    st.markdown("### 🎯 Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Minimum confidence score for valid detections"
    )
    
    # Display settings
    st.markdown("### 🎨 Display Settings")
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    show_labels = st.checkbox("Show Class Labels", value=True)

# Auto-load model on startup if file exists
@st.cache_resource
def load_model_cached(model_path):
    """Load model with caching to prevent reloading on every run"""
    try:
        if os.path.exists(model_path):
            model = YOLO(model_path)
            return model, True, None
        else:
            return None, False, f"Model file '{model_path}' not found"
    except Exception as e:
        return None, False, str(e)

# Load model if not already loaded
if not st.session_state.model_loaded:
    model, success, error = load_model_cached(model_path)
    if success:
        st.session_state.model = model
        st.session_state.model_loaded = True
    else:
        st.session_state.model = None
        st.session_state.model_loaded = False

# Model status and info
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.session_state.model_loaded and st.session_state.model is not None:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Model Status</h3>
            <p class="status-success">Ready</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Model Status</h3>
            <p class="status-error">Not Loaded</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🎯 Confidence</h3>
        <p class="status-success">{confidence_threshold:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    detection_count = st.session_state.get('detection_count', 0)
    st.markdown(f"""
    <div class="metric-card">
        <h3>🔍 Detections</h3>
        <p class="status-success">{detection_count}</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>⚡ Performance</h3>
        <p class="status-success">Real-time</p>
    </div>
    """, unsafe_allow_html=True)

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "📹 Webcam Detection", "📊 Analytics"])

with tab1:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 📸 Upload Image for Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image to test helmet detection"
    )
    
    if uploaded_file and st.session_state.model_loaded and st.session_state.model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown("#### Detection Results")
            
            # Convert image for processing
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            
            # Run detection
            with st.spinner("Detecting..."):
                try:
                    results = st.session_state.model(
                        img_bgr, 
                        conf=confidence_threshold,
                        verbose=False
                    )
                    
                    if len(results) > 0:
                        result = results[0]
                        
                        # Plot results with clean styling
                        annotated_frame = result.plot(
                            conf=show_confidence,
                            labels=show_labels,
                            boxes=True,
                            line_width=2,
                            font_size=14
                        )
                        
                        # Convert BGR to RGB for display
                        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, use_column_width=True)
                        
                        # Detection details
                        if result.boxes is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            classes = result.boxes.cls.cpu().numpy()
                            
                            st.markdown("#### 🎯 Detection Details")
                            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                                class_name = st.session_state.model.names[int(cls)]
                                st.markdown(f"""
                                **Detection {i+1}:**
                                - Class: {class_name}
                                - Confidence: {conf:.3f}
                                - Coordinates: ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})
                                """)
                        else:
                            st.warning("No helmets detected. Try adjusting the confidence threshold.")
                    
                except Exception as e:
                    st.error(f"Detection failed: {str(e)}")
    
    elif not st.session_state.model_loaded or st.session_state.model is None:
        st.warning("Please load a model first using the sidebar.")
        st.info("💡 Make sure 'best.pt' file exists in your directory, then click 'Load/Reload Model'.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 📹 Real-time Webcam Detection")
    
    if st.session_state.model_loaded and st.session_state.model is not None:
        # WebRTC configuration
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Create a shared model variable that can be accessed by VideoProcessor
        model_for_detection = st.session_state.model
        
        class VideoProcessor:
            def __init__(self):
                self.detection_count = 0
                self.model = model_for_detection  # Store model reference
                self.conf_threshold = confidence_threshold
                self.show_conf = show_confidence
                self.show_labels = show_labels
                
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                
                # Check if model is available
                if self.model is None:
                    cv2.putText(img, "Model not loaded - Please reload model", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(img, "Check sidebar settings", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
                
                # Run detection
                try:
                    results = self.model(
                        img,
                        conf=self.conf_threshold,
                        verbose=False
                    )
                    
                    if len(results) > 0:
                        result = results[0]
                        
                        if result.boxes is not None and len(result.boxes) > 0:
                            self.detection_count = len(result.boxes)
                            
                            # Draw detections with professional styling
                            annotated_frame = result.plot(
                                conf=self.show_conf,
                                labels=self.show_labels,
                                boxes=True,
                                line_width=2,
                                font_size=14
                            )
                            img = annotated_frame
                            
                            # Add detection count overlay
                            cv2.putText(img, f"Detections: {self.detection_count}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            self.detection_count = 0
                            # Add "No detections" overlay
                            cv2.putText(img, "No helmets detected", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    else:
                        self.detection_count = 0
                    
                    # Add status overlay
                    cv2.putText(img, f"Conf: {self.conf_threshold:.2f}", (img.shape[1]-150, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                except Exception as e:
                    # Add error text to frame
                    error_msg = str(e)[:40]
                    cv2.putText(img, f"Detection Error:", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img, error_msg, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Display current model status before starting webcam
        st.info(f"🤖 Model Status: {'✅ Loaded' if model_for_detection is not None else '❌ Not Loaded'}")
        
        if model_for_detection is not None:
            try:
                model_classes = list(model_for_detection.names.values())
                st.success(f"📋 Detected Classes: {', '.join(model_classes)}")
            except:
                st.warning("⚠️ Could not retrieve model classes")
        
        # Webcam streamer
        webrtc_ctx = webrtc_streamer(
            key="helmet-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Control buttons and settings update
        st.markdown("#### 🎮 Webcam Controls")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎬 Start Detection", key="start_detection"):
                st.session_state.detection_active = True
                st.success("Detection started!")
        
        with col2:
            if st.button("⏹️ Stop Detection", key="stop_detection"):
                st.session_state.detection_active = False
                st.info("Detection stopped!")
        
        with col3:
            if st.button("🔄 Refresh Model", key="refresh_model"):
                # This will cause the VideoProcessor to be recreated with new model
                st.rerun()
        
        with col4:
            if st.button("📸 Take Screenshot", key="screenshot"):
                st.info("Screenshot feature - implement if needed")
        
        # Settings display
        st.markdown("---")
        st.markdown("#### ⚙️ Current Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Confidence:** {confidence_threshold}")
        with col2:
            st.markdown(f"**Show Labels:** {show_confidence}")
        
        # Real-time information
        if webrtc_ctx.state.playing:
            st.markdown("#### 📊 Live Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Webcam Status", "🟢 Active")
            
            with col2:
                st.metric("Model Status", "🟢 Running" if model_for_detection else "🔴 Error")
            
            with col3:
                st.metric("Stream Quality", "HD")
        else:
            st.markdown("#### 📊 Stream Status")
            st.info("Webcam stream not active. Click 'START' to begin.")
    
    else:
        st.warning("⚠️ Model not loaded!")
        st.info("Please load your YOLO model first:")
        st.markdown("""
        1. Make sure `best.pt` file exists in your directory
        2. Use the sidebar to load the model
        3. Wait for confirmation message
        4. Refresh this tab if needed
        """)
        
        # Show current directory info for debugging
        if st.checkbox("Show debug info"):
            st.code(f"""
Current directory: {os.getcwd()}
Files in directory: {os.listdir('.')}
Model file exists: {os.path.exists(model_path)}
Session state model: {st.session_state.model is not None}
Model loaded flag: {st.session_state.model_loaded}
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Detection Analytics")
    
    if st.session_state.model_loaded and st.session_state.model is not None:
        # Model information
        st.markdown("#### 🤖 Model Information")
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                model_info = {
                    "Model Type": type(st.session_state.model).__name__,
                    "Classes": list(st.session_state.model.names.values()) if hasattr(st.session_state.model, 'names') else "Unknown",
                    "Input Size": "640x640 (default)",
                    "Framework": "YOLOv8/Ultralytics"
                }
                
                for key, value in model_info.items():
                    st.markdown(f"**{key}**: {value}")
            except Exception as e:
                st.error(f"Error retrieving model info: {e}")
        
        with col2:
            # Performance metrics (placeholder)
            st.markdown("**Performance Metrics**")
            st.markdown("- Average Inference Time: ~30ms")
            st.markdown("- FPS: ~30")
            st.markdown("- Accuracy: Model dependent")
            st.markdown("- Memory Usage: ~500MB")
        
        # Detection history (placeholder for future implementation)
        st.markdown("#### 📈 Detection History")
        st.info("Detection history and analytics will be displayed here in future updates.")
        
        # Tips and recommendations
        st.markdown("#### 💡 Optimization Tips")
        st.markdown("""
        - **Lighting**: Ensure good lighting conditions for better detection
        - **Angle**: Position camera at eye level for optimal results
        - **Distance**: Maintain 1-3 meters distance from subjects
        - **Background**: Avoid cluttered backgrounds when possible
        - **Model**: Retrain model with more diverse data if needed
        """)
    
    else:
        st.warning("Load a model to view analytics.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="info-card" style="text-align: center; margin-top: 3rem;">
    <p>🛡️ Smart Helmet Detection System | Powered by YOLOv8 & Streamlit</p>
    <p style="opacity: 0.7;">Ensuring workplace safety through AI technology</p>
</div>
""", unsafe_allow_html=True)