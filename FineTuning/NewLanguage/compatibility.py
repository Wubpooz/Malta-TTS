# Fix import for TTS in python>=3.11
import sys

try:
    import TTS
    import trainer
    import coqpit
except ImportError:
    import coqui_tts
    sys.modules["TTS"] = coqui_tts
    import coqui_tts_trainer
    sys.modules["trainer"] = coqui_tts_trainer
    import coqpit_config
    sys.modules["coqpit"] = coqpit_config
