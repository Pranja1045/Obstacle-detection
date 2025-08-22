import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- OPTIMIZED Frame Processing Function ---
def process_frame(frame):
    # --- PERFORMANCE OPTIMIZATION 1: Resize the frame ---
    # Processing smaller frames is much faster.
    RESIZE_WIDTH = 480
    height, width, _ = frame.shape
    scale = RESIZE_WIDTH / width
    resized_height = int(height * scale)
    frame = cv2.resize(frame, (RESIZE_WIDTH, resized_height))

    StepSize = 5
    img = frame.copy()
    blur = cv2.bilateralFilter(img, 9, 40, 40)
    edges = cv2.Canny(blur, 50, 100)
    img_h, img_w, _ = img.shape

    EdgeArray = []
    # --- PERFORMANCE OPTIMIZATION 2: Vectorized Edge Finding ---
    # This replaces the slow nested Python loops with a fast NumPy operation.
    for j in range(0, img_w, StepSize):
        column = edges[:, j] # Get the entire column
        indices = np.where(column > 0)[0] # Find all non-zero pixel indices
        if indices.size > 0:
            # Get the last (lowest) index, which is the edge point
            pixel = (j, indices[-1])
        else:
            # If no edge is found, default to the bottom of the frame
            pixel = (j, img_h - 1)
        EdgeArray.append(pixel)

    if len(EdgeArray) < 3:
        return frame, edges

    num_chunks = 3
    chunks = np.array_split(EdgeArray, num_chunks)
    
    avg_points = []
    for chunk in chunks:
        if chunk.size > 0:
            x_vals = [pt[0] for pt in chunk]
            y_vals = [pt[1] for pt in chunk]
            avg_x = int(np.average(x_vals))
            avg_y = int(np.average(y_vals))
            avg_points.append((avg_x, avg_y))
            cv2.line(frame, (img_w // 2, img_h), (avg_x, avg_y), (255, 0, 0), 2)

    if len(avg_points) < num_chunks:
        return frame, edges

    left_point, forward_point, right_point = avg_points
    
    direction = "Path: FORWARD"
    color = (0, 255, 0)

    # Obstacle detection threshold adjusted for the resized frame
    if forward_point[1] < (img_h * 0.7):
        if left_point[1] > right_point[1]:
            direction = "Obstacle: Turn LEFT"
        else:
            direction = "Obstacle: Turn RIGHT"
        color = (0, 0, 255)

        box_center = forward_point
        box_size = 100 # Box size adjusted for smaller frame
        top_left = (box_center[0] - box_size // 2, box_center[1] - box_size // 2)
        bottom_right = (box_center[0] + box_size // 2, box_center[1] + box_size // 2)
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        cv2.putText(frame, "OBSTACLE", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.putText(frame, direction, (img_w // 2 - 120, img_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                
    return frame, edges

# --- WebRTC Video Transformer ---
class ObstacleDetector(VideoTransformerBase):
    def __init__(self):
        self.show_edges = False

    def set_show_edges(self, show_edges):
        self.show_edges = show_edges

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_frame, edges_frame = process_frame(img)
        
        # The output frame of the transformer must be the same size as the input.
        # We resize our processed frame back to the original size.
        original_height, original_width, _ = img.shape
        if self.show_edges:
            # Convert edges to color so it can be resized properly
            edges_color = cv2.cvtColor(edges_frame, cv2.COLOR_GRAY2BGR)
            return cv2.resize(edges_color, (original_width, original_height))
        else:
            return cv2.resize(processed_frame, (original_width, original_height))

# --- Streamlit App UI ---
st.set_page_config(page_title="Live Obstacle Detection", layout="wide")

st.title("🤖 Obstacle Detection and Path Planning")
st.caption("Process a live webcam feed or upload a video file.")
st.sidebar.title("Made by Mahi Priyadarshi ❤️")
st.sidebar.header("Configuration")

source_option = st.sidebar.selectbox("Select Input Source", ["Webcam", "Upload Video"])

if source_option == "Webcam":
    st.sidebar.subheader("Webcam Settings")
    show_edges = st.sidebar.checkbox("Show Canny Edges", value=False)
    
    ctx = webrtc_streamer(
        key="obstacle-detection",
        video_processor_factory=ObstacleDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if ctx.video_processor:
        ctx.video_processor.set_show_edges(show_edges)

elif source_option == "Upload Video":
    st.sidebar.subheader("Video Upload Settings")
    uploaded_file = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])
    run = st.sidebar.button("▶️ Start Processing")

    if run and uploaded_file:
        video_source = None
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_source = tfile.name
        
        if video_source:
            cap = cv2.VideoCapture(video_source)
            frame_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.info("Video processing finished.")
                    break
                
                processed_frame, _ = process_frame(frame)
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(processed_frame_rgb, channels="RGB")

            cap.release()
            if os.path.exists(video_source):
                os.remove(video_source)
