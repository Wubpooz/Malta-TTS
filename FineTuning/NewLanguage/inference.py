import compatibility

import gc
import torch
import os
import torchaudio
from tqdm import tqdm
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from utils import split_into_sentences, check_and_split_by_limit


def load_model(LORA_trained: bool, xtts_checkpoint: str, model, config, checkpoint_dir, xtts_vocab, use_deepspeed):
  print("Loading checkpoint...")
  checkpoint = torch.load(xtts_checkpoint, map_location="cpu")
  is_lora = any("lora_A" in k or "lora_B" in k for k in checkpoint.keys())
  if LORA_trained or is_lora:
    print("Detected LoRA adapter weights. Loading as LoRA model.")
    lora_config = LoraConfig(
      r=8,
      lora_alpha=16,
      target_modules=["c_attn", "c_proj"],
      lora_dropout=0.05,
      bias="none",
      task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, lora_config)
    model.load_state_dict(checkpoint, strict=False)
  else:
    print("Detected standard model weights. Loading as base model.")
    model.load_checkpoint(
      config,
      checkpoint_dir=checkpoint_dir,
      checkpoint_path=xtts_checkpoint,
      vocab_path=xtts_vocab,
      use_deepspeed=use_deepspeed,
      eval=True
    )
  
  return model


def text_processing(tts_text, lang_code, model):
  try:
    sentences = split_into_sentences(tts_text, lang_code)
    print(f"Split into {len(sentences)} sentences.")
    tts_texts = check_and_split_by_limit(sentences, char_limit=model.tokenizer.char_limits.get(lang_code, 400), lang_code=lang_code)
  except:
    tts_texts = [tts_text]

  print(f"Final text chunks: {len(tts_texts)}")
  for i, text in enumerate(tts_texts):
    print(f"  Chunk {i+1}: {len(text)} chars - '{text[:50]}{'...' if len(text) > 50 else ''}'")
  return tts_texts  



def inference(xtts_checkpoint, xtts_config, xtts_vocab, tts_text, speaker_audio_file, lang_code, temperature=0.7, length_penalty=1.0, repetition_penalty=10.0, top_k=50, top_p=0.8, LORA_trained=False):
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
  model = load_model(LORA_trained, xtts_checkpoint, model, config, checkpoint_dir, xtts_vocab, use_deepspeed)

  if not hasattr(model.tokenizer, "char_limits"):
    model.tokenizer.char_limits = {}
  if lang_code not in model.tokenizer.char_limits:
    model.tokenizer.char_limits[lang_code] = model.tokenizer.char_limits.get("en", 400)
    print(f"Added char_limits for {lang_code} language.")

  model.to(device)
  print("Model loaded successfully!")

  if lang_code not in ["en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko"]: #default XTTS supported languages
    from utils import add_language_to_tokenizer
    add_language_to_tokenizer(model.tokenizer, lang_code)
    print("Applied custom tokenizer.")


  print("Computing speaker latents...")
  gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_audio_file])
  print("Speaker latents computed successfully!")

  print("Processing text...")
  tts_texts = text_processing(tts_text, lang_code, model)

  wav_chunks = []
  print("Running inference...")
  for i, text in enumerate(tqdm(tts_texts, desc="Processing sentences")):
    if not text.strip():
      continue  # Skip empty sentences

    #TODO can use inference_stream
    try:
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
    except Exception as e:
      print(f"Warning: Failed to synthesize chunk {i+1}: '{text[:50]}...' - Error: {e}")
      continue
  
  print("Inference successful!")

  del model, gpt_cond_latent, speaker_embedding, config
  torch.cuda.empty_cache()
  gc.collect()

  if len(wav_chunks) == 0:
    raise RuntimeError("No audio chunks were successfully generated!")
  elif len(wav_chunks) > 1:
    # Add small silence between chunks for better flow
    silence = torch.zeros(int(0.1 * 24000))  # 0.1 second silence at 24kHz
    final_chunks = []
    for i, chunk in enumerate(wav_chunks):
      final_chunks.append(chunk)
      if i < len(wav_chunks) - 1:  # Don't add silence after last chunk
        final_chunks.append(silence)
    return torch.cat(final_chunks, dim=0).unsqueeze(0)
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
  os.makedirs(os.path.dirname(output_file), exist_ok=True)
  torchaudio.save(output_file, audio_waveform, sample_rate=24000)
  print(f"Audio saved to {output_file}")