import streamlit as st
import cv2
import numpy as np

def process_frame(frame):
    
    StepSize = 5

    img = frame.copy()

    blur = cv2.bilateralFilter(img, 9, 40, 40)
    edges = cv2.Canny(blur, 50, 100)

    img_h, img_w, _ = img.shape

    EdgeArray = []
    for j in range(0, img_w, StepSize):
        pixel = (j, 0)
        for i in range(img_h - 10, 0, -1):
            if edges.item(i, j) == 255:
                pixel = (j, i)
                break
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
        st.warning("Not all path segments were detected clearly.")
        return frame, edges

    left_point, forward_point, right_point = avg_points
    
    direction = "Path: FORWARD"
    color = (0, 255, 0)

    if forward_point[1] < (img_h * 0.7): 
        obstacle_detected = True
        if left_point[1] > right_point[1]:
            direction = "Obstacle: Turn LEFT"
        else:
            direction = "Obstacle: Turn RIGHT"
        color = (0, 0, 255) 

        box_center = forward_point
        box_size = 150
        top_left = (box_center[0] - box_size // 2, box_center[1] - box_size // 2)
        bottom_right = (box_center[0] + box_size // 2, box_center[1] + box_size // 2)
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        cv2.putText(frame, "OBSTACLE", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(frame, direction, (img_w // 2 - 150, img_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                
    return frame, edges



st.set_page_config(page_title="Live Obstacle Detection", layout="wide")

st.title("🤖 Live Obstacle Detection and Path Planning")
st.caption("This app uses OpenCV to detect a clear path from a live camera feed.")
st.sidebar.title("Made by Mahi Priyadarshi ❤️ ")

st.sidebar.header("Configuration")
camera_source = st.sidebar.selectbox("Select Camera Source", ("Use Webcam (0)", "Use External Cam (1)"), index=0)
camera_index = 0 if camera_source == "Use Webcam (0)" else 1

show_edges = st.sidebar.checkbox("Show Canny Edges", value=False)

st.sidebar.markdown("---")
run = st.sidebar.button("▶️ Start Camera")
stop = st.sidebar.button("⏹️ Stop Camera")
st.sidebar.markdown("---")

if 'is_running' not in st.session_state:
    st.session_state.is_running = False

if run:
    st.session_state.is_running = True
if stop:
    st.session_state.is_running = False

col1, col2 = st.columns(2)
frame_placeholder = col1.empty()
if show_edges:
    edges_placeholder = col2.empty()
else:
    col2.empty() 

if st.session_state.is_running:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error(f"Error: Could not open camera at index {camera_index}.")
    else:
        st.success("Camera started successfully! Streaming...")
        while st.session_state.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame from camera. Stream might have ended.")
                break

            processed_frame, edges_frame = process_frame(frame)
            
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(processed_frame_rgb, channels="RGB")
            if show_edges:
                edges_placeholder.image(edges_frame, caption="Canny Edges")

        cap.release()
        if not stop: 
             st.info("Stream ended. Press 'Start Camera' to run again.")

elif not st.session_state.is_running:
    frame_placeholder.info("Camera is off. Press 'Start Camera' in the sidebar to begin streaming.")

cv2.destroyAllWindows()