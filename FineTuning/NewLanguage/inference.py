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
  model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=use_deepspeed, eval=True)
  model.to(device)
  print("Model loaded successfully!")

  if lang_code not in ["en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko"]:
    from utils import add_language_to_VoiceBpeTokenizer
    add_language_to_VoiceBpeTokenizer(lang_code)
    print("Applied custom tokenizer.")


  print("Computing speaker latents...")
  gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=[speaker_audio_file],
    # gpt_cond_len=model.config.gpt_cond_len,
    # max_ref_length=model.config.max_ref_len,
    # sound_norm_refs=model.config.sound_norm_refs,
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
    # print(f"Processing sentence {i+1}: {text[:50]}...")

    out = model.inference(
      text=text,
      language=lang_code,
      gpt_cond_latent=gpt_cond_latent,
      speaker_embedding=speaker_embedding,
      enable_text_splitting=True,
      temperature=temperature,
      length_penalty=length_penalty,
      repetition_penalty=repetition_penalty,
      top_k=top_k,
      top_p=top_p,
      speed=1.0
    )
    wav_chunks.append(torch.tensor(out["wav"]))
    # wav_chunks.append(torch.tensor(out["wav"]).unsqueeze(0))
    rate = 216000 #TODO
    audio_time = len(torch.tensor(out["wav"]).unsqueeze(0) / rate)
    print(f"Audio time for sentence {i+1}: {audio_time:.2f} seconds")

  print("Inference successful!")


  # combined_wav = torch.cat(wav_chunks, dim=1)

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