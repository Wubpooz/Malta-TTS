# Fix import for TTS in python>=3.11
import sys

try:
    import TTS
    import trainer
except ImportError:
    import coqui_tts
    sys.modules["TTS"] = coqui_tts
    import coqui_tts_trainer
    sys.modules["trainer"] = coqui_tts_trainer
