#py -3.10 -m pip install -r requirements.txt
#streamlit run xtts_app.py

import streamlit as st
from TTS.api import TTS
import torch
import os
import contextlib
import io
import time

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
import torch.serialization

# Skip license prompt
os.environ["COQUI_TOS_AGREED"] = "1"

# Fix torch deserialization globals
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

st.set_page_config(page_title="Maltese XTTS", layout="centered")

st.markdown(
  """
  <style>
  .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px 0;
    font-size: 0.875rem;
    z-index: 100;
    border-top: 1px solid #ccc;
  }

  @media (prefers-color-scheme: light) {
    .footer {
      background-color: #f0f2f6;
      color: #333;
      border-color: #e0e0e0;
    }
  }

  @media (prefers-color-scheme: dark) {
    .footer {
      background-color: #0e1117;
      color: #ccc;
      border-color: #222;
    }
    .footer a {
      color: #61dafb;
    }
  }

  .footer a {
    text-decoration: none;
  }
  </style>

  <div class="footer">
    © 2025 Mathieu Waharte — <a href="https://github.com/Wubpooz/Malta-TTS" target="_blank">View on GitHub</a>
  </div>
  """,
  unsafe_allow_html=True
)

st.title("🗣️ XTTS v2 - Maltese Text-to-Speech")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.markdown(f"**Using device:** `{device}`")

# UI inputs
speaker_wav = "test_female_maltese_8s.mp3"  # or use file_uploader()
text_input = st.text_area("Enter Maltese Text", "Merħba f'Malta! It-temp illum huwa sabiħ ħafna.")
inference_lang = st.selectbox("Inference language", ["ar", "it", "en", "fr", "sp"], index=0)
output_path = "output_maltese.wav"

if st.button("Generate Speech") and speaker_wav and text_input.strip():
  log_output_placeholder = st.empty()
  captured_logs = io.StringIO()
  start_time_load = time.time()
  with st.spinner("Loading XTTS model... (This may take a few minutes with the initial download of the model)"):
    try:
      with contextlib.redirect_stdout(captured_logs):
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
        end_time_load = time.time()
        elapsed_time_load = round(end_time_load - start_time_load, 2)
        st.success(f"✅ Model loaded! Time elapsed: {elapsed_time_load}s")
        if captured_logs.getvalue():
          st.code("Model Loading Logs:\n" + captured_logs.getvalue())
    except Exception as e:
      st.error(f"Failed to load XTTS model: {e}")
      st.stop()

  captured_logs.truncate(0)
  captured_logs.seek(0)

  start_time_synth = time.time()
  with st.spinner("Generating speech..."):
    try:
      with contextlib.redirect_stdout(captured_logs):
        tts.tts_to_file(
          text=text_input,
          speaker_wav=speaker_wav,
          language=inference_lang,
          file_path=output_path
        )
      end_time_synth = time.time()
      elapsed_time_synth = round(end_time_synth - start_time_synth, 2)
      st.success(f"✅ Synthesis complete! Time elapsed: {elapsed_time_synth}s")
      audio_file = open(output_path, "rb")
      st.audio(audio_file.read(), format="audio/wav")
      if captured_logs.getvalue():
        st.code("Synthesis Logs:\n" + captured_logs.getvalue())
    except Exception as e:
      st.error(f"Error during synthesis: {e}")
      st.stop()
