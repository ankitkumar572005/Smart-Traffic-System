import streamlit as st
import os
import subprocess
import shutil
from main import process_video
from detector import VehicleDetector

@st.cache_resource
def load_detector():
    """Load the model once and keep it in cache for speed and RAM safety."""
    return VehicleDetector()

st.set_page_config(page_title="Smart Traffic System", page_icon="🚦", layout="centered")

st.title("🚦 Smart Traffic Monitoring System")
st.markdown("""
Upload a traffic video to analyze vehicle counts, bounding boxes, license plates, and helmet safety status!
""")

# Setup directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)

st.markdown("### Choose an input method:")

tab1, tab2, tab3 = st.tabs(["Upload File", "YouTube Link", "Demo Video"])

run_analysis = False
input_path = None

with tab1:
    uploaded_file = st.file_uploader("Upload a traffic video (.mp4)", type=["mp4"])
    if uploaded_file is not None:
        st.video(uploaded_file)
        if st.button("Run AI Analysis", type="primary", key="file_btn"):
            input_path = os.path.join("uploads", uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            run_analysis = True

with tab2:
    yt_url = st.text_input("Paste a YouTube Video Link:")
    if yt_url:
        st.video(yt_url)
        st.info("Note: To protect server resources, only the first 20 seconds will be analyzed!")
        if st.button("Run AI on YouTube Clip", type="primary", key="yt_btn"):
            with st.spinner("Downloading 20-second clip from YouTube..."):
                target_out = "uploads/youtube_vid.mp4"
                if os.path.exists(target_out): os.remove(target_out)
                
                try:
                    import yt_dlp
                    # Attempt 1: Safe 20s clipped download
                    ydl_opts = {
                        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
                        'outtmpl': target_out,
                        'download_ranges': yt_dlp.utils.download_range_func(None, [(0, 20)]),
                        'force_keyframes_at_cuts': True,
                        'quiet': True
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([yt_url])
                except Exception:
                    # Attempt 2: Fallback to full download (capped at 50MB) if ffmpeg slicing fails for Shorts
                    if os.path.exists(target_out): os.remove(target_out)
                    try:
                        ydl_opts_fallback = {
                            'format': 'best[ext=mp4]/best', 
                            'max_filesize': 50 * 1024 * 1024,
                            'outtmpl': target_out,
                            'noplaylist': True,
                            'quiet': True
                        }
                        with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl_fb:
                            ydl_fb.download([yt_url])
                    except Exception as e_fallback:
                        st.error(f"Download heavily failed: {e_fallback}")
                    
                if os.path.exists(target_out):
                    input_path = target_out
                    run_analysis = True
                else:
                    st.error("Download failed to create MP4.")

with tab3:
    st.info("🎥 Don't have a traffic video handy? Run the AI on the built-in demo track!")
    if os.path.exists("sample_traffic.mp4"):
        if st.button("Run AI on Demo Video", type="primary", key="demo_btn"):
            input_path = "sample_traffic.mp4"
            run_analysis = True

if run_analysis and input_path:
    with st.spinner("Processing video through YOLOv8 and DeepSORT..."):
        output_mp4v_path = os.path.join("output", "annotated_raw.mp4")
        output_h264_path = os.path.join("output", "annotated_playable.mp4")
        csv_path = os.path.join("output", "traffic_log.csv")
        
        # Clean old runs
        if os.path.exists(output_mp4v_path): os.remove(output_mp4v_path)
        if os.path.exists(output_h264_path): os.remove(output_h264_path)
        if os.path.exists(csv_path): os.remove(csv_path)

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Run inference using the CACHED detector with High-Stability Error Trapping
        try:
            detector = load_detector()
            success = process_video(input_path, output_mp4v_path, csv_path, detector, progress_bar, status_text)
        except Exception as e:
            st.error("⚠️ SYSTEM REBOOT REQUIRED")
            st.warning(f"""
            This application is currently running on a Limited 1GB Free Cloud Server. 
            The AI process was interrupted to prevent a total crash. 
            
            **Technical Note for Recruiters:** This illustrates the 'Memory-Wall' constraint in free-tier deployment. 
            In a production environment (AWS/GCP), this would be resolved with a higher RAM instance.
            
            **Error Details:** {e}
            """)
            success = False
        
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
