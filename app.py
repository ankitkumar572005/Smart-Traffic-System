import streamlit as st
import os
import subprocess
import shutil
from main import process_video

st.set_page_config(page_title="Smart Traffic System", page_icon="🚦", layout="centered")

st.title("🚦 Smart Traffic Monitoring System")
st.markdown("""
Upload a traffic video to analyze vehicle counts, bounding boxes, license plates, and helmet safety status!
""")

# Setup directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)

uploaded_file = st.file_uploader("Upload a traffic video (.mp4)", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Run AI Analysis", type="primary"):
        with st.spinner("Processing video through YOLOv8 and DeepSORT..."):
            
            # Save uploaded file safely
            input_path = os.path.join("uploads", uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            output_mp4v_path = os.path.join("output", "annotated_raw.mp4")
            output_h264_path = os.path.join("output", "annotated_playable.mp4")
            csv_path = os.path.join("output", "traffic_log.csv")
            
            # Clean old runs
            if os.path.exists(output_mp4v_path): os.remove(output_mp4v_path)
            if os.path.exists(output_h264_path): os.remove(output_h264_path)
            if os.path.exists(csv_path): os.remove(csv_path)

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. Run inference
            success = process_video(input_path, output_mp4v_path, csv_path, progress_bar, status_text)
            
            if success:
                status_text.text("Converting video to H264 for web playback...")
                
                # 2. Convert standard OpenCV mp4v to h264 for web browser using ffmpeg
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", output_mp4v_path, "-vcodec", "libx264", "-acodec", "aac", output_h264_path],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    status_text.text("Conversion complete!")
                    
                    st.success("Analysis Finished Successfully!")
                    
                    st.subheader("Results")
                    # Display playable video
                    if os.path.exists(output_h264_path):
                        st.video(output_h264_path)
                    
                    # Display Download Buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if os.path.exists(output_h264_path):
                            with open(output_h264_path, "rb") as f:
                                st.download_button("⬇️ Download Annotated Video", f.read(), file_name="annotated_traffic.mp4", mime="video/mp4")
                    with col2:
                        if os.path.exists(csv_path):
                            with open(csv_path, "rb") as f:
                                st.download_button("⬇️ Download CSV Log", f.read(), file_name="traffic_log.csv", mime="text/csv")
                                
                    # Optionally display CSV dump
                    import pandas as pd
                    if os.path.exists(csv_path):
                        try:
                            df = pd.read_csv(csv_path)
                            if not df.empty:
                                st.dataframe(df)
                        except:
                            pass
                            
                except Exception as e:
                    status_text.text(f"Error converting video: {e}. Raw clip is still available.")
                    with open(output_mp4v_path, "rb") as f:
                        st.download_button("⬇️ Download Raw Video (.mp4v)", f.read(), file_name="annotated_raw.mp4")
            else:
                st.error("Failed to process the video.")
