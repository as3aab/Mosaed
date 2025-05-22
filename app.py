
import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os
import pandas as pd

st.set_page_config(page_title="Whisper Transcriber", layout="wide")
st.title("🎙 Whisper Audio Transcriber (Faster-Whisper Version)")

uploaded_files = st.file_uploader("Upload one or more audio/video files", type=["mp3", "mp4", "wav", "m4a"], accept_multiple_files=True)

if uploaded_files:
    model_size = st.selectbox("Select model size", ["tiny", "base", "small", "medium"])
    st.write("⏳ Initializing model...")
    model = WhisperModel(model_size, compute_type="int8")

    for uploaded_file in uploaded_files:
        st.markdown(f"### 🔍 Processing: `{uploaded_file.name}`")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        segments, _ = model.transcribe(tmp_file_path)
        results = []
        for segment in segments:
            results.append({
                "Start": round(segment.start, 2),
                "End": round(segment.end, 2),
                "Text": segment.text.strip()
            })

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        csv_filename = f"{uploaded_file.name}_transcription.csv"
        csv_path = os.path.join(tempfile.gettempdir(), csv_filename)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        with open(csv_path, "rb") as f:
            st.download_button(label="📥 Download Transcription CSV", data=f, file_name=csv_filename, mime="text/csv")
