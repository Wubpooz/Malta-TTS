import torch
import os
import torchaudio
from tqdm import tqdm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def inference(xtts_checkpoint, xtts_config, xtts_vocab, tts_text, speaker_audio_file, lang_code, temperature=0.7, length_penalty=1.0, repetition_penalty=10.0, top_k=50, top_p=0.8):
  """Run inference using the XTTS model with the provided configuration and text.
  Args:
      xtts_checkpoint (str): Path to the XTTS model checkpoint.
      xtts_config (str): Path to the XTTS configuration file.
      xtts_vocab (str): Path to the XTTS vocabulary file.
      tts_text (str): Text to be synthesized.
      speaker_audio_file (str): Path to the audio file of the speaker for conditioning.
      lang (str): Language code for the text. Supported languages include "en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko", "mt".
  Returns:
      torch.Tensor: Synthesized audio waveform.
  """

  checkpoint_dir = os.path.dirname(xtts_checkpoint)
  speakers_file = os.path.join(checkpoint_dir, "speakers_xtts.pth")
  if not os.path.exists(speakers_file):
    # Create an empty speakers file
    torch.save({}, speakers_file)



  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  try:
    import deepspeed # type: ignore
    use_deepspeed = device.startswith("cuda")  # Use deepspeed only if CUDA is available
  except ImportError:
    use_deepspeed = False
    print("Deepspeed is not installed, using CPU/GPU without deepspeed.")

  print("Loading model...")
  config = XttsConfig()
  config.load_json(xtts_config)
  print("Config loaded.")
  
  print("Initializing model...")
  model = Xtts.init_from_config(config)
  
  print("Loading checkpoint...")
  model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=use_deepspeed)
  model.to(device)
  print("Model loaded successfully!")

  if lang_code not in ["en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko"]:
    import TTS.tts.layers.xtts.tokenizer as tokenizer
    import re

    _original_preprocess_text = tokenizer.VoiceBpeTokenizer.preprocess_text

    def custom_preprocess_text(self, txt, lang):
      if lang == lang_code:
        # transliterate ?
        txt = txt.lower()
        txt = re.sub(re.compile(r"\s+"), " ", txt)
        return txt.strip()
      return _original_preprocess_text(self, txt, lang)

    # Monkey-patch
    tokenizer.VoiceBpeTokenizer.preprocess_text = custom_preprocess_text
    print("Applied custom tokenizer.")


  print("Computing speaker latents...")
  gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=[speaker_audio_file],
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, # type: ignore
    max_ref_length=XTTS_MODEL.config.max_ref_len, # type: ignore
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs, # type: ignore
  )
  print("Speaker latents computed successfully!")

  # Split text into sentences for better quality
  # print("Processing text...")
  # try:
  #   import nltk
  #   from nltk.data import find
  #   try:
  #     find('tokenizers/punkt')
  #   except LookupError:
  #     print("NLTK 'punkt' tokenizer not found. Downloading...")
  #     nltk.download('punkt')
  #     print("NLTK 'punkt' tokenizer downloaded successfully.")
    
  #   from nltk.tokenize import sent_tokenize
  #   tts_texts = sent_tokenize(tts_text)
  #   print(f"Split into {len(tts_texts)} sentences.")
  # except ImportError:
  #   print("NLTK not available, processing as single text.")
  #   tts_texts = [tts_text]

  tts_texts = [tts_text]


  wav_chunks = []
  print("Running inference...")
  for i, text in enumerate(tqdm(tts_texts, desc="Processing sentences")):
    print(f"Processing sentence {i+1}: {text[:50]}...")

    out = model.inference(
      text=text,
      language=lang_code,
      gpt_cond_latent=gpt_cond_latent,
      speaker_embedding=speaker_embedding,
      temperature=temperature,
      length_penalty=length_penalty,
      repetition_penalty=repetition_penalty,
      top_k=top_k,
      top_p=top_p
    )
    wav_chunks.append(torch.tensor(out["wav"]))

  print("Inference successful!")

  if len(wav_chunks) > 1:
    return torch.cat(wav_chunks, dim=0).unsqueeze(0)
  else:
    return torch.tensor(wav_chunks[0]).unsqueeze(0)


if __name__ == "__main__":
  from parsers import create_inference_parser
  parser = create_inference_parser()
  args = parser.parse_args()
  
  print("Starting inference...")
  audio_waveform = inference(
    xtts_checkpoint=args.xtts_checkpoint,
    xtts_config=args.xtts_config,
    xtts_vocab=args.xtts_vocab,
    tts_text=args.tts_text,
    speaker_audio_file=args.speaker_audio_file,
    lang_code=args.lang,
    temperature=args.temperature,
    length_penalty=args.length_penalty,
    repetition_penalty=args.repetition_penalty,
    top_k=args.top_k,
    top_p=args.top_p
  )
  print("Inference completed!")

  output_file = args.output_file if args.output_file else "output.wav"
  torchaudio.save(output_file, audio_waveform, sample_rate=24000)
  print(f"Audio saved to {output_file}")