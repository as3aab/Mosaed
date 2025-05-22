import streamlit as st
import whisper
import os
import tempfile
import pandas as pd
from datetime import timedelta

st.set_page_config(page_title="Whisper Transcriber", layout="centered")

# واجهة ثنائية اللغة
lang = st.sidebar.selectbox("🌐 اختر اللغة / Choose Language", ["العربية", "English"])

LABELS = {
    "العربية": {
        "title": "📄 أداة تفريغ الفيديوهات باستخدام Whisper",
        "upload": "📤 ارفع ملفات الفيديو أو الصوت",
        "processing": "جاري التفريغ...",
        "done": "✅ تم التفريغ! يمكنك تحميل الملف:",
        "download": "⬇️ تحميل ملف CSV",
        "more_files": "📂 أضف ملفات أخرى"
    },
    "English": {
        "title": "📄 Video/Audio Transcriber using Whisper",
        "upload": "📤 Upload your audio or video files",
        "processing": "Transcribing...",
        "done": "✅ Done! Download your transcript:",
        "download": "⬇️ Download CSV",
        "more_files": "📂 Add more files"
    }
}

st.title(LABELS[lang]["title"])
uploaded_files = st.file_uploader(LABELS[lang]["upload"], type=["mp4", "mp3", "m4a", "wav"], accept_multiple_files=True)

if uploaded_files:
    model = whisper.load_model("large")
    for uploaded_file in uploaded_files:
        st.info(f'{LABELS[lang]["processing"]} ({uploaded_file.name})')
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        result = model.transcribe(tmp_path, word_timestamps=False)
        segments = result["segments"]

        data = []
        for seg in segments:
            start = str(timedelta(seconds=round(seg["start"], 2)))
            end = str(timedelta(seconds=round(seg["end"], 2)))
            text = seg["text"]
            data.append((start, end, text))

        df = pd.DataFrame(data, columns=["Start", "End", "Text"])
        csv_file = f"{uploaded_file.name}_transcription.csv"
        df.to_csv(csv_file, index=False)

        with open(csv_file, "rb") as f:
            st.success(f'{LABELS[lang]["done"]} {uploaded_file.name}')
            st.download_button(label=LABELS[lang]["download"], data=f, file_name=csv_file, mime="text/csv")

    st.button(LABELS[lang]["more_files"])