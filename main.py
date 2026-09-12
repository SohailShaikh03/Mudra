import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import string
import os
import threading
from io import BytesIO
from collections import deque
from gtts import gTTS

# Optional cloud WebRTC support for Web/Mobile browser camera
try:
    import av
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        RTCConfiguration,
        WebRtcMode,
    )
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

# Optional translator
try:
    from googletrans import Translator
    translator = Translator()
except Exception:
    translator = None

# Optional TensorFlow / Keras (Falls back to pure NumPy weights for lightning performance)
try:
    from tensorflow import keras
except Exception:
    keras = None

# ==========================================
# PAGE CONFIGURATION & SIGNATURE THEME
# ==========================================
st.set_page_config(
    page_title="Mudra: ISL Real-Time Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Signature Cyber-Saffron & Obsidian CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Fira+Code:wght@400;600&display=swap');

    /* Global Typography & Background Glow */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #151D2E 0%, #0B0F19 75%);
        color: #F3F4F6;
    }

    /* Branded Header Styling */
    .brand-hero {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(6, 182, 212, 0.08) 100%);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    
    .brand-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FBBF24 0%, #F59E0B 50%, #EF4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-sub {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 6px;
        font-weight: 400;
    }

    .author-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 10px;
    }

    /* Status Pill Chips */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Glassmorphic Surface Cards */
    .glass-card {
        background: rgba(22, 31, 48, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
    }

    /* Sentence Studio Container */
    .sentence-display-box {
        background: #0D131F;
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 12px;
        padding: 18px 22px;
        min-height: 80px;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        font-family: 'Fira Code', monospace;
        font-size: 1.35rem;
        color: #F8FAFC;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4);
    }

    .letter-token {
        background: rgba(245, 158, 11, 0.2);
        border: 1px solid #F59E0B;
        color: #FDE68A;
        border-radius: 6px;
        padding: 2px 10px;
        font-weight: 600;
    }

    .space-token {
        background: rgba(255, 255, 255, 0.08);
        color: #64748B;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.85rem;
    }

    /* Streamlit Button Tweaks & Touch Targets */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        min-height: 48px !important;
        transition: all 0.2s ease-in-out !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3) !important;
    }

    /* Primary Accent Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.35) !important;
    }

    /* Responsive Mobile Adjustments */
    @media (max-width: 768px) {
        .brand-title {
            font-size: 1.85rem;
        }
        .brand-hero {
            padding: 16px 18px;
        }
        .sentence-display-box {
            font-size: 1.15rem;
            min-height: 65px;
        }
        div.stButton > button {
            min-height: 50px !important;
            font-size: 0.95rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL & MEDIAPIPE INITIALIZATION
# ==========================================
@st.cache_resource(show_spinner="Initializing ISL Deep Learning Engine...")
def load_model_isl():
    """Loads the pre-trained Indian Sign Language model (NumPy weights or Keras)."""
    # 1. Fast path: Pure NumPy weights (lightweight, instant, zero TensorFlow required)
    if os.path.exists("isl_weights.npz"):
        try:
            weights = np.load("isl_weights.npz")
            return {"type": "numpy", "weights": weights}, "isl_weights.npz"
        except Exception:
            pass

    # 2. Fallback: Keras model if TensorFlow is installed
    if keras is not None:
        candidates = ["model_isl_fixed.h5", "model_isl.h5"]
        for path in candidates:
            if os.path.exists(path):
                try:
                    loaded = keras.models.load_model(path)
                    return {"type": "keras", "model": loaded}, path
                except Exception:
                    pass
    return None, None

model, loaded_model_path = load_model_isl()

# Initialize MediaPipe Hands
try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
except Exception:
    try:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    except Exception:
        import mediapipe.solutions.hands as mp_hands
        import mediapipe.solutions.drawing_utils as mp_drawing
        import mediapipe.solutions.drawing_styles as mp_drawing_styles

# ISL Alphabet definition: 9 numbers ('1'-'9') + 26 uppercase letters ('A'-'Z') = 35 classes
ISL_ALPHABET = [str(i) for i in range(1, 10)] + list(string.ascii_uppercase)

# ==========================================
# SESSION STATE MANAGEMENT
# ==========================================
if "sentence" not in st.session_state:
    st.session_state.sentence = []
if "last_committed_char" not in st.session_state:
    st.session_state.last_committed_char = None
if "voice_language" not in st.session_state:
    st.session_state.voice_language = "English"
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "local_camera_running" not in st.session_state:
    st.session_state.local_camera_running = False
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.65
if "stability_frames_req" not in st.session_state:
    st.session_state.stability_frames_req = 8

# ==========================================
# UTILITY ALGORITHMS & PREPROCESSING
# ==========================================
def calc_landmark_list(image, landmarks):
    """Converts MediaPipe normalized landmarks to pixel coordinates."""
    image_width, image_height = image.shape[1], image.shape[0]
    return [[int(lm.x * image_width), int(lm.y * image_height)] for lm in landmarks.landmark]

def pre_process_landmark(landmarks):
    """Normalizes landmarks to be translation and scale invariant (42 numerical features)."""
    base_x, base_y = landmarks[0]
    relative_landmarks = [[x - base_x, y - base_y] for x, y in landmarks]
    flat_landmarks = np.array(relative_landmarks, dtype=np.float32).flatten()
    
    max_val = np.max(np.abs(flat_landmarks))
    if max_val == 0:
        max_val = 1.0
        
    normalized_landmarks = flat_landmarks / max_val
    return normalized_landmarks.tolist()

def predict_sign(landmark_vector):
    """Predicts character and returns (predicted_char, confidence_score)."""
    if model is None:
        return "-", 0.0
    try:
        if isinstance(model, tuple):
            loaded_obj = model[0]
        else:
            loaded_obj = model

        if isinstance(loaded_obj, dict) and loaded_obj.get("type") == "numpy":
            w = loaded_obj["weights"]
            x = np.array([landmark_vector], dtype=np.float32)
            h1 = np.maximum(0, np.dot(x, w['w1']) + w['b1'])
            h2 = np.maximum(0, np.dot(h1, w['w2']) + w['b2'])
            logits = np.dot(h2, w['w3']) + w['b3']
            exp_logits = np.exp(logits - np.max(logits))
            preds = (exp_logits / np.sum(exp_logits))[0]
        else:
            kmodel = loaded_obj["model"] if isinstance(loaded_obj, dict) else loaded_obj
            df = pd.DataFrame([landmark_vector])
            preds = kmodel.predict(df, verbose=0)[0]

        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        return ISL_ALPHABET[idx], conf
    except Exception:
        return "-", 0.0

def translate_and_speak(text, lang_name):
    """Translates text if needed and synthesizes audio via gTTS."""
    if not text.strip():
        return None, text
    
    lang_map = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
    lang_code = lang_map.get(lang_name, "en")
    
    translated_text = text
    if lang_code != "en" and translator is not None:
        try:
            res = translator.translate(text, dest=lang_code)
            translated_text = res.text
        except Exception:
            translated_text = text

    try:
        tts = gTTS(text=translated_text, lang=lang_code, slow=False)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read(), translated_text
    except Exception:
        return None, translated_text

# ==========================================
# WEBRTC VIDEO PROCESSOR (CLOUD / PHONE / BROWSER)
# ==========================================
class ISLWebRtcProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.hands = mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.recent_predictions = deque(maxlen=10)
        self.current_char = "-"
        self.current_conf = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        char_detected = "-"
        conf_detected = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate coordinates and run inference
                landmark_list = calc_landmark_list(img, hand_landmarks)
                norm_vector = pre_process_landmark(landmark_list)
                char_detected, conf_detected = predict_sign(norm_vector)

                # Draw glowing custom skeleton
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 158, 11), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(6, 182, 212), thickness=2)
                )

                # HUD Overlay on frame
                cv2.rectangle(img, (15, 15), (210, 85), (15, 23, 42), -1)
                cv2.rectangle(img, (15, 15), (210, 85), (245, 158, 11), 2)
                cv2.putText(img, f"{char_detected}", (30, 70), cv2.FONT_HERSHEY_DUPLEX, 1.8, (245, 158, 11), 3)
                cv2.putText(img, f"{int(conf_detected*100)}%", (125, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (6, 182, 212), 2)

        # Thread-safe update of current state
        with self.lock:
            self.current_char = char_detected
            self.current_conf = conf_detected
            if char_detected != "-" and conf_detected >= 0.65:
                self.recent_predictions.append(char_detected)
            else:
                self.recent_predictions.append(None)

        return av.VideoFrame.from_ndarray(img, format="bgr24") if HAS_WEBRTC else frame

# ==========================================
# SIDEBAR CONFIGURATION & CONTROLS
# ==========================================
with st.sidebar:
    # Mudra Branding & Logo
    if os.path.exists("mudralogo.jpg"):
        st.image("mudralogo.jpg", use_container_width=True)
    
    st.markdown("### ⚙️ System Settings")
    
    # Camera Mode Switcher
    camera_mode_options = []
    if HAS_WEBRTC:
        camera_mode_options.append("🌐 Web Browser Camera (Cloud / Phone)")
    camera_mode_options.append("💻 Local Camera (OpenCV)")
    
    chosen_camera_mode = st.selectbox(
        "Camera Mode",
        options=camera_mode_options,
        index=0,
        help="Select 'Web Browser Camera' for hosting on Streamlit Community Cloud and mobile devices. Use 'Local Camera' for direct hardware access on your PC."
    )
    
    st.markdown("---")
    
    # Speech Synthesis Language
    st.session_state.voice_language = st.selectbox(
        "🗣️ Speech Output Voice",
        options=["English", "Hindi", "Marathi"],
        index=0
    )
    
    st.markdown("---")
    
    # Sensitivity & Stability Adjustments
    with st.expander("🎛️ Gesture Tuning", expanded=False):
        st.session_state.confidence_threshold = st.slider(
            "Confidence Cutoff",
            min_value=0.50,
            max_value=0.95,
            value=0.65,
            step=0.05,
            help="Higher values reduce false positives."
        )
        st.session_state.stability_frames_req = st.slider(
            "Stability Hold (Frames)",
            min_value=4,
            max_value=16,
            value=8,
            step=1,
            help="Number of consecutive stable frames required before registering a letter."
        )

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748B; font-size: 0.8rem;">
        Mudra ISL Engine • v2.0 Signature Edition<br>
        Crafted by <b style="color: #FBBF24;">Sohail Shaikh</b>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# HERO BRAND BANNER
# ==========================================
st.markdown("""
<div class="brand-hero">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
        <div>
            <h1 class="brand-title">🤟 MUDRA</h1>
            <div class="brand-sub">Real-Time Indian Sign Language Translation & Speech Synthesis System</div>
            <div class="author-pill">⚡ Engineered by Sohail Shaikh</div>
        </div>
        <div style="display: flex; flex-direction: column; gap: 8px; align-items: flex-end;">
            <div class="status-badge">🟢 Neural Engine: Active</div>
            <div style="font-size: 0.8rem; color: #94A3B8;">35 Hand Gestures Supported (A-Z, 1-9)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MAIN WORKSPACE LAYOUT
# ==========================================
main_col_left, main_col_right = st.columns([3, 2], gap="large")

# LEFT COLUMN: CAMERA FEED
with main_col_left:
    st.markdown("### 📷 Live Camera Feed")
    
    if "Web Browser Camera" in chosen_camera_mode and HAS_WEBRTC:
        st.caption("📱 WebRTC Stream enabled — Works seamlessly on Chrome, Safari, mobile smartphones & PCs.")
        
        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        webrtc_ctx = webrtc_streamer(
            key="mudra-isl-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "video": {
                    "facingMode": "user",
                    "width": {"ideal": 640},
                    "height": {"ideal": 480}
                },
                "audio": False
            },
            video_processor_factory=ISLWebRtcProcessor,
            async_processing=True,
        )

        # Pull predictions from processor
        if webrtc_ctx.video_processor:
            with webrtc_ctx.video_processor.lock:
                history = list(webrtc_ctx.video_processor.recent_predictions)
            
            # Check stability in sliding window
            if len(history) >= st.session_state.stability_frames_req:
                valid_items = [x for x in history if x is not None and x != "-"]
                if valid_items:
                    most_common = max(set(valid_items), key=valid_items.count)
                    if valid_items.count(most_common) >= (st.session_state.stability_frames_req - 2):
                        if not st.session_state.sentence or st.session_state.sentence[-1] != most_common:
                            st.session_state.sentence.append(most_common)
                            st.session_state.last_committed_char = most_common
                            st.toast(f"Captured: {most_common}", icon="✨")

    else:
        # LOCAL OPENCV CAMERA MODE
        st.caption("💻 Direct Hardware Stream via OpenCV (For Local Testing)")
        cam_start_col, cam_stop_col = st.columns(2)
        
        start_pressed = cam_start_col.button(
            "▶️ Start Camera", 
            key="start_local_cam",
            use_container_width=True,
            disabled=st.session_state.local_camera_running
        )
        stop_pressed = cam_stop_col.button(
            "⏹️ Stop Camera",
            key="stop_local_cam",
            use_container_width=True,
            disabled=not st.session_state.local_camera_running
        )

        if start_pressed:
            st.session_state.local_camera_running = True
            st.rerun()

        if stop_pressed:
            st.session_state.local_camera_running = False
            st.rerun()

        video_frame_placeholder = st.empty()
        
        if st.session_state.local_camera_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access local webcam. Please check hardware permissions.")
                st.session_state.local_camera_running = False
            else:
                consecutive_count = 0
                candidate_char = None
                
                with mp_hands.Hands(
                    model_complexity=0,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7
                ) as hands:
                    while st.session_state.local_camera_running and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Webcam stream disconnected.")
                            break

                        frame = cv2.flip(frame, 1)
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(img_rgb)

                        current_pred = "-"
                        current_conf = 0.0

                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                landmark_list = calc_landmark_list(frame, hand_landmarks)
                                norm_vector = pre_process_landmark(landmark_list)
                                current_pred, current_conf = predict_sign(norm_vector)

                                mp_drawing.draw_landmarks(
                                    frame,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 158, 11), thickness=2, circle_radius=3),
                                    mp_drawing.DrawingSpec(color=(6, 182, 212), thickness=2)
                                )

                                # HUD banner on frame
                                cv2.rectangle(frame, (15, 15), (210, 85), (15, 23, 42), -1)
                                cv2.rectangle(frame, (15, 15), (210, 85), (245, 158, 11), 2)
                                cv2.putText(frame, f"{current_pred}", (30, 70), cv2.FONT_HERSHEY_DUPLEX, 1.8, (245, 158, 11), 3)
                                cv2.putText(frame, f"{int(current_conf*100)}%", (125, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (6, 182, 212), 2)

                                # Stability debouncing
                                if current_pred != "-" and current_conf >= st.session_state.confidence_threshold:
                                    if current_pred == candidate_char:
                                        consecutive_count += 1
                                    else:
                                        candidate_char = current_pred
                                        consecutive_count = 1

                                    if consecutive_count >= st.session_state.stability_frames_req:
                                        if not st.session_state.sentence or st.session_state.sentence[-1] != candidate_char:
                                            st.session_state.sentence.append(candidate_char)
                                        consecutive_count = 0
                                else:
                                    consecutive_count = 0
                                    candidate_char = None

                        video_frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                cap.release()
        else:
            video_frame_placeholder.markdown("""
            <div style="background: rgba(17, 24, 39, 0.6); border: 2px dashed rgba(245, 158, 11, 0.3); border-radius: 14px; padding: 60px 20px; text-align: center;">
                <div style="font-size: 3rem;">📹</div>
                <h4 style="color: #FBBF24; margin-top: 12px;">Camera is Idle</h4>
                <p style="color: #94A3B8; max-width: 400px; margin: 0 auto;">Click <b>Start Camera</b> above to initiate real-time gesture recognition.</p>
            </div>
            """, unsafe_allow_html=True)

# RIGHT COLUMN: SENTENCE STUDIO & CONTROLS
with main_col_right:
    st.markdown("### ✍️ Sentence Studio")
    
    # Render constructed sentence
    sentence_tokens = st.session_state.sentence
    if sentence_tokens:
        tokens_html = ""
        for token in sentence_tokens:
            if token == " ":
                tokens_html += '<span class="space-token">␣ SPACE</span>'
            else:
                tokens_html += f'<span class="letter-token">{token}</span>'
        sentence_str = "".join(sentence_tokens)
    else:
        tokens_html = '<span style="color: #64748B; font-style: italic;">No gestures recognized yet. Make a sign in front of the camera...</span>'
        sentence_str = ""

    st.markdown(f"""
    <div class="sentence-display-box">
        {tokens_html}
    </div>
    """, unsafe_allow_html=True)
    
    # Action Toolbar
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    if btn_col1.button("␣ Space", use_container_width=True):
        st.session_state.sentence.append(" ")
        st.rerun()

    if btn_col2.button("⌫ Backspace", use_container_width=True):
        if st.session_state.sentence:
            st.session_state.sentence.pop()
            st.rerun()

    if btn_col3.button("🗑️ Clear All", use_container_width=True):
        st.session_state.sentence = []
        st.rerun()

    st.markdown("---")

    # Speech Synthesis & Translation Studio
    st.markdown("### 🔊 Voice & Translation Studio")
    audio_col1, audio_col2 = st.columns([1, 1])

    speak_clicked = audio_col1.button("🎙️ Speak Sentence", kind="primary", use_container_width=True)
    copy_clicked = audio_col2.button("📋 Copy Text", use_container_width=True)

    if copy_clicked and sentence_str:
        st.toast(f"Copied to clipboard: \"{sentence_str}\"", icon="📋")

    if speak_clicked:
        if sentence_str.strip():
            with st.spinner(f"Generating audio in {st.session_state.voice_language}..."):
                audio_bytes, translated_output = translate_and_speak(sentence_str, st.session_state.voice_language)
                if audio_bytes:
                    st.success(f"🗣️ Speaking in {st.session_state.voice_language}: **{translated_output}**")
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                else:
                    st.warning("Audio synthesis could not complete. Check internet connection.")
        else:
            st.warning("Sentence is empty. Record some signs before speaking!")

# ==========================================
# INTERACTIVE ISL REFERENCE CHEATSHEET
# ==========================================
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
with st.expander("📖 Indian Sign Language (ISL) Interactive Alphabet Cheatsheet", expanded=False):
    st.markdown("""
    Learn the 35 recognized gestures. Ensure your hand is well-lit, clearly facing the camera, and positioned within the frame.
    """)
    
    tab_signs, tab_poster, tab_tips = st.tabs(["🔤 Alphabet & Digits Grid", "🖼️ Official ISL Poster", "💡 Best Recognition Tips"])
    
    with tab_signs:
        digits = [str(i) for i in range(1, 10)]
        letters = list(string.ascii_uppercase)
        
        st.markdown("**Numbers (1 to 9):**")
        d_cols = st.columns(9)
        for i, d in enumerate(digits):
            d_cols[i].markdown(f"<div style='text-align:center; padding: 10px; background: rgba(245, 158, 11, 0.15); border-radius: 8px; font-weight:700; color: #FBBF24;'>{d}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("**Alphabet (A to Z):**")
        
        # 13 columns per row for 26 letters
        row1_cols = st.columns(13)
        for i, char in enumerate(letters[:13]):
            row1_cols[i].markdown(f"<div style='text-align:center; padding: 8px; background: rgba(6, 182, 212, 0.15); border-radius: 8px; font-weight:700; color: #38BDF8;'>{char}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        row2_cols = st.columns(13)
        for i, char in enumerate(letters[13:]):
            row2_cols[i].markdown(f"<div style='text-align:center; padding: 8px; background: rgba(6, 182, 212, 0.15); border-radius: 8px; font-weight:700; color: #38BDF8;'>{char}</div>", unsafe_allow_html=True)

    with tab_poster:
        if os.path.exists("Poster_isl.png"):
            st.image("Poster_isl.png", caption="Indian Sign Language Gesture Chart", use_container_width=True)
        else:
            st.info("Poster image 'Poster_isl.png' not found.")

    with tab_tips:
        st.markdown("""
        - **Lighting Matters:** Ensure your hand is front-lit and not back-lit by a window or lamp.
        - **Hand Stability:** Hold each sign firmly for ~0.5 seconds to trigger the stability hold engine.
        - **Frame Position:** Keep your hand centered in the frame, 1 to 2 feet away from the lens.
        - **Transitions:** Move your hand smoothly between signs to prevent intermediate transitional frames.
        """)

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 24px; color: #64748B; font-size: 0.85rem; border-top: 1px solid rgba(255, 255, 255, 0.08);">
    <b>Mudra: ISL Translator</b> • Engineered by <b>Sohail Shaikh</b><br>
    Powered by MediaPipe Hands, TensorFlow, and Streamlit Community Cloud
</div>
""", unsafe_allow_html=True)
