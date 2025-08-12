import torch
import os
from tqdm import tqdm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def inference(xtts_checkpoint, xtts_config, xtts_vocab, tts_text, speaker_audio_file, lang):
  """Run inference using the XTTS model with the provided configuration and text.
  Args:
      xtts_checkpoint (str): Path to the XTTS model checkpoint.
      xtts_config (str): Path to the XTTS configuration file.
      xtts_vocab (str): Path to the XTTS vocabulary file.
      tts_text (str): Text to be synthesized.
      speaker_audio_file (str): Path to the audio file of the speaker for conditioning.
      lang (str): Language code for the text. Supported languages include "en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko".
  Returns:
      torch.Tensor: Synthesized audio waveform.
  """

  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  try:
    import deepspeed # type: ignore
    use_deepspeed = device.startswith("cuda")  # Use deepspeed only if CUDA is available
  except ImportError:
    use_deepspeed = False
    print("Deepspeed is not installed, using CPU/GPU without deepspeed.")

  config = XttsConfig()
  print("Loading config...")
  config.load_json(xtts_config)
  print("Config Loaded.")
  print("Initing model...")
  XTTS_MODEL = Xtts.init_from_config(config)
  print("Model Init, loadign checkpoint...")
  checkpoint_dir = os.path.dirname(xtts_checkpoint)
  XTTS_MODEL.load_checkpoint(config, speaker_file_path="", checkpoint_dir=checkpoint_dir, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=use_deepspeed)
  XTTS_MODEL.to(device)
  print("Model loaded successfully!")


  gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, # type: ignore
    max_ref_length=XTTS_MODEL.config.max_ref_len, # type: ignore
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs, # type: ignore
  )

  # import nltk
  # from nltk.data import find

  # try:
  #   find('tokenizers/punkt')
  # except LookupError:
  #   print("NLTK 'punkt' tokenizer not found. downloading it now... (you can also download it manually using \"python -c \"import nltk; nltk.download('punkt')\"\")")
  #   nltk.download('punkt')
  #   print("NLTK 'punkt' tokenizer downloaded successfully.")
  #   pass

  # from nltk.tokenize import sent_tokenize
  # tts_texts = sent_tokenize(tts_text)

  import TTS.tts.layers.xtts.tokenizer as tokenizer
  import re

  _original_preprocess_text = tokenizer.VoiceBpeTokenizer.preprocess_text

  def custom_preprocess_text(self, txt, lang):
      if lang == "mt":  # Maltese
          txt = txt.lower()
          txt = re.sub(re.compile(r"\s+"), " ", txt)
          # transliterate ?
          return txt.strip()
      return _original_preprocess_text(self, txt, lang)

  # Monkey-patch
  tokenizer.VoiceBpeTokenizer.preprocess_text = custom_preprocess_text


  wav_chunks = []
  print("Infering...")
  # for text in tqdm(tts_texts):
  wav_chunk = XTTS_MODEL.inference(
    text=tts_text,
    language=lang,
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    temperature=float(XTTS_MODEL.config.temperature), # default 0.1
    length_penalty=float(XTTS_MODEL.config.length_penalty), # default 1.0
    repetition_penalty=float(XTTS_MODEL.config.repetition_penalty), # default 10.0
    top_k=int(XTTS_MODEL.config.top_k), # default 10
    top_p=float(XTTS_MODEL.config.top_p), # default 0.3
  )
  wav_chunks.append(torch.tensor(wav_chunk["wav"]))
  print("Inference successful!")

  return torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()



if __name__ == "__main__":
  from parsers import create_inference_parser
  parser = create_inference_parser()
  args = parser.parse_args()
  audio_waveform = inference(
    xtts_checkpoint=args.xtts_checkpoint,
    xtts_config=args.xtts_config,
    xtts_vocab=args.xtts_vocab,
    tts_text=args.tts_text,
    speaker_audio_file=args.speaker_audio_file,
    lang=args.lang
  )

  print("Inference completed. Audio waveform shape:", audio_waveform.shape)

  import torchaudio
  output_file = args.output_file if args.output_file else "output.wav"
  torchaudio.save(output_file, audio_waveform, sample_rate=24000)
  print(f"Audio saved to {output_file}")

  # try:
  #   from IPython.display import Audio
  #   Audio(output_file, autoplay=True)
  # except ImportError:
  #   print("IPython.display.Audio not available. You can play the audio file using any audio player.")