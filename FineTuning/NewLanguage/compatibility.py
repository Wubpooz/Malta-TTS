# Fix import for TTS in python>=3.11
import sys

try:
    import TTS
except ImportError:
    import coqui_tts
    sys.modules["TTS"] = coqui_tts
