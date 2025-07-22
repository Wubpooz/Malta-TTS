#py -3.10 -m pip install -r requirements.txt
#streamlit run xtts_app.py

import streamlit as st
import torch
import os
from TTS.api import TTS
from TTS.utils.manage import ModelManager

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
import torch.serialization

# Skip license prompt
os.environ["COQUI_TOS_AGREED"] = "1"

# Fix torch deserialization globals
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])


@st.cache_resource
def download_xtts_model():
  manager = ModelManager()
  return manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")


st.set_page_config(page_title="Maltese XTTS", layout="centered")
st.title("🗣️ XTTS v2 - Maltese Text-to-Speech")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.markdown(f"**Using device:** `{device}`")

# UI inputs
speaker_wav = "test_female_maltese_8s.mp3"  # or use file_uploader()
text_input = st.text_area("Enter Maltese Text", "Merħba f'Malta! It-temp illum huwa sabiħ ħafna.")
inference_lang = st.selectbox("Inference language", ["ar", "it", "en", "fr", "sp"], index=0)
output_path = "output_maltese.wav"

if st.button("Generate Speech") and speaker_wav and text_input.strip():
  download_xtts_model()
  with st.spinner("Loading XTTS model..."):
    try:
      tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
    except Exception as e:
      st.error(f"Failed to load XTTS model: {e}")
      st.stop()

  with st.spinner("Synthesizing..."):
    try:
      tts.tts_to_file(
        text=text_input,
        speaker_wav=speaker_wav,
        language=inference_lang,
        file_path=output_path
      )
      st.success("✅ Synthesis complete!")
      audio_file = open(output_path, "rb")
      st.audio(audio_file.read(), format="audio/wav")
    except Exception as e:
      st.error(f"Error during synthesis: {e}")
      st.stop()