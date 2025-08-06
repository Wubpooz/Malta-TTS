# -*- coding: utf-8 -*-
"""maltese_xtts.py
This script is used to fine-tune the XTTS model for Maltese language and run inference on it.
It includes functions for training the model and performing inference, as well as a command-line interface to
control the process.
Based on https://github.com/daswer123/xtts-webui/blob/main/scripts/utils/gpt_train.py & https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages/blob/main/train_gpt_xtts.py
"""

# Inference imports
import torch
import torchaudio
from tqdm import tqdm

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


# Training imports
import os
import gc
import argparse

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager


# Tokenizer imports
import json
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


# ========================== Data preparation ==========================
#TODO




# ========================== Download ==========================
# CHECKPOINTS_OUT_PATH = os.path.join(output_path, "models", f"{version}")
#   os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


#   # DVAE files
#   DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
#   MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
#   DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
#   MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

#   if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
#     print(" > Downloading DVAE files!")
#     ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


#   TOKENIZER_FILE_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/vocab.json"
#   XTTS_CHECKPOINT_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/model.pth"
#   # if useful, can add config and speakers files
#   # XTTS_CONFIG_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/config.json"
#   # XTTS_SPEAKER_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/speakers_xtts.pth"

#   XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))
#   TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))

#   if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
#     print(f" > Downloading XTTS v{version} files!")
#     ModelManager._download_model_files([TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True) # private API

#   # if useful, can transfer files to ready folder
#   # READY_MODEL_PATH = os.path.join(output_path,"ready")
#   # if not os.path.exists(READY_MODEL_PATH):
#   #   os.makedirs(READY_MODEL_PATH)
#   # NEW_TOKENIZER_FILE = os.path.join(READY_MODEL_PATH, "vocab.json")
#   # shutil.copy(TOKENIZER_FILE, NEW_TOKENIZER_FILE)
#   # TOKENIZER_FILE = NEW_TOKENIZER_FILE # vocab.json file

#   if custom_model:
#     if os.path.isfile(custom_model) and custom_model.endswith('.pth'):
#         XTTS_CHECKPOINT = custom_model
#         print(f" > Loading custom model: {XTTS_CHECKPOINT}")
#     else:
#         raise ValueError(f"Error: The specified custom model is not a valid .pth file: {custom_model}")
def download(output_path: str, version: str = "main", custom_model: str = ""):
  """Download the XTTS model files and prepare the environment for training or inference.

  Args:
      output_path (str): Path to the output directory. They will be saved in a subdirectory named "models/<version>" within this path.
      version (str): Version of the XTTS model to download. Default is "main".
      custom_model (str): Path to a custom model checkpoint (.pth file) to use instead of the default XTTS model.
  Raises:
      ValueError: If the custom model is not a valid .pth file.
  """
  CHECKPOINTS_OUT_PATH = os.path.join(output_path, "models", f"{version}")
  os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)
  
  DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
  MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
  DVAE_CHECKPOINT = os.path.join(output_path, os.path.basename(DVAE_CHECKPOINT_LINK))
  MEL_NORM_FILE = os.path.join(output_path, os.path.basename(MEL_NORM_LINK))

  if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], output_path, progress_bar=True)


  TOKENIZER_FILE_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/vocab.json"
  XTTS_CHECKPOINT_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/model.pth"
  XTTS_CONFIG_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/config.json"

  XTTS_CHECKPOINT = os.path.join(output_path, os.path.basename(XTTS_CHECKPOINT_LINK))
  TOKENIZER_FILE = os.path.join(output_path, os.path.basename(TOKENIZER_FILE_LINK))

  if custom_model:
    if os.path.isfile(custom_model) and custom_model.endswith('.pth'):
      XTTS_CHECKPOINT = custom_model
      print(f" > Loading custom model: {XTTS_CHECKPOINT}")
    else:
      raise ValueError(f"Error: The specified custom model is not a valid .pth file: {custom_model}")
    

  print(f" > Downloading XTTS v{version} files!")
  if not os.path.isfile(XTTS_CHECKPOINT):  # don't download again if the checkpoint exists or when using a custom model
    print(" > Downloading XTTS checkpoint...")
    ModelManager._download_model_files([XTTS_CHECKPOINT_LINK], output_path, progress_bar=True) # private API
  if not os.path.isfile(TOKENIZER_FILE):
    print(" > Downloading XTTS tokenizer...")
    ModelManager._download_model_files([TOKENIZER_FILE_LINK], output_path, progress_bar=True)
  if not os.path.isfile(XTTS_CONFIG_LINK):
    print(" > Downloading XTTS config file...")
    ModelManager._download_model_files([XTTS_CONFIG_LINK], output_path, progress_bar=True)
  print(" > XTTS model files downloaded successfully!")

  # if useful, can transfer files to ready folder
  # READY_MODEL_PATH = os.path.join(output_path,"ready")
  # if not os.path.exists(READY_MODEL_PATH):
  #   os.makedirs(READY_MODEL_PATH)
  # NEW_TOKENIZER_FILE = os.path.join(READY_MODEL_PATH, "vocab.json")
  # shutil.copy(TOKENIZER_FILE, NEW_TOKENIZER_FILE)
  # TOKENIZER_FILE = NEW_TOKENIZER_FILE # vocab.json file

  return MEL_NORM_FILE, DVAE_CHECKPOINT, XTTS_CHECKPOINT, TOKENIZER_FILE





# ========================== Tokenizer Extension ==========================
def merge_tokenizers_preserve_ids(old_tokenizer_path, new_tokenizer_path, output_path):
    """
    Merges two vocabularies, preserving the token IDs from the old tokenizer
    and adding new tokens from the new tokenizer.
    """
    print(f"Merging tokenizers from {old_tokenizer_path} and {new_tokenizer_path} into {output_path}")
    with open(os.path.join(old_tokenizer_path, 'vocab.json')) as f:
      old_vocab = json.load(f)
      print(f"Old tokenizer vocabulary size: {len(old_vocab)}")
    with open(os.path.join(new_tokenizer_path, 'vocab.json')) as f:
      new_vocab = json.load(f)
      print(f"New tokenizer vocabulary size: {len(new_vocab)}")

    combined_vocab = old_vocab.copy()
    new_id = max(old_vocab.values())
    
    # Iterate through the new vocabulary and add words not in the old one
    for word, _ in new_vocab.items():
      if word not in combined_vocab:
        new_id += 1
        combined_vocab[word] = new_id

    print(f"Combined vocabulary size: {len(combined_vocab)}")
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'vocab.json'), 'w') as fp:
      json.dump(combined_vocab, fp, ensure_ascii=False, indent=2)
    print(f"Combined vocabulary saved to {os.path.join(output_path, 'vocab.json')}")

    return combined_vocab


def extend_tokenizer(output_path: str, metadata_path: str, language: str, extended_vocab_size: int = 100000):
  """Extends the XTTS tokenizer with new vocabulary from the provided metadata file.
  This function combines the existing tokenizer with a new tokenizer trained on the provided metadata.
  It saves the new tokenizer in a specified directory and updates the vocabulary to include new tokens.
  Args:
      output_path (str): Path to the output directory where the tokenizer files will be saved.
      metadata_path (str): Path to the metadata file containing training data.
      language (str): Language code for the new language to be added.
      extended_vocab_size (int): Desired size of the extended vocabulary. Default is 100000.
  """
  root = os.path.join(output_path, "")
  
  old_tokenizer_path = os.path.join(root, "old_tokenizer/")
  new_tokenizer_path = os.path.join(root, "new_tokenizer/")
  merged_tokenizer_path = os.path.join(root, "merged_tokenizer/")

  os.makedirs(old_tokenizer_path, exist_ok=True)
  existing_tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
  existing_tokenizer.model.save(old_tokenizer_path)
  print(f"Original tokenizer loaded with {len(existing_tokenizer.get_vocab())} tokens.")

  traindf = pd.read_csv(metadata_path, sep="|")
  texts = traindf.text.to_list()
  new_tokenizer = Tokenizer(BPE())
  new_tokenizer.pre_tokenizer = Whitespace() # type: ignore
  trainer = BpeTrainer(special_tokens=[f"[{language}]"], vocab_size=extended_vocab_size) # type: ignore
  print(f"Training new tokenizer with {len(texts)} texts...")
  new_tokenizer.train_from_iterator(iter(texts), trainer=trainer)
  new_tokenizer.add_special_tokens([f"[{language}]"])

  print(f"New tokenizer trained with {len(new_tokenizer.get_vocab())} tokens.")
  os.makedirs(new_tokenizer_path, exist_ok=True)
  new_tokenizer.model.save(new_tokenizer_path)

  merge_tokenizers_preserve_ids(old_tokenizer_path, new_tokenizer_path, merged_tokenizer_path)

  # 4. Now, create the final tokenizer by combining the merged vocab with the new merges.txt
  merged_vocab_file = os.path.join(merged_tokenizer_path, 'vocab.json')
  new_merges_file = os.path.join(new_tokenizer_path, 'merges.txt')

  final_tokenizer = Tokenizer(BPE.from_files(vocab=merged_vocab_file, merges=new_merges_file)) # type: ignore
  final_tokenizer.pre_tokenizer = Whitespace() # type: ignore
  final_tokenizer.add_special_tokens([f"[{language}]"])

  # 5. Overwrite the original vocab.json with the new, extended one
  final_tokenizer.save(os.path.join(root, "vocab.json"))

  # Clean up temporary files
  os.system(f'rm -rf {old_tokenizer_path} {new_tokenizer_path} {merged_tokenizer_path}')

  print(f"Tokenizer has been successfully extended and saved to {os.path.join(root, 'vocab.json')}")



def adjust_config(output_path: str, version: str, language: str):
  """Adjust the XTTS configuration file to include the new language.
  Args:
      output_path (str): Path to the output directory where the config file is located (it will be appended with "/models/{version}/config.json"). 
      version (str): Version of the XTTS model.
      language (str): Language code for the new language to be added.
  """
  config_path = os.path.join(output_path, "models", f"{version}", "/config.json")
  with open(config_path, "r") as f:
    config = json.load(f)
  config["languages"] += [language]
  with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)



# ========================== Training ==========================
def train_gpt(metadatas, num_epochs=100, batch_size=3, grad_acumm=84, output_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"), lr=5e-06, weight_decay=1e-2, save_step=10000, custom_model="", version="main", max_text_length=200, max_audio_length=255995, multi_gpu=False):
  """Train the GPT XTTS model for Maltese language.
  This function sets up the training configuration, downloads necessary files, initializes the model, and starts the training process.
  It also saves the final model checkpoint and configuration files after training.

  Args:
      metadatas (list): A list of metadata strings in the format "train_csv_path,eval_csv_path,language".
      num_epochs (int): Number of epochs for training. Default is 100.
      batch_size (int): Mini batch size. Default is 3.
      grad_acumm (int): Gradient accumulation steps. Default is 84.
      output_path (str): Path to save the model checkpoints and outputs. Default is the current directory/"checkpoints".
      lr (float): Learning rate for the optimizer. Default is 5e-6.
      weight_decay (float): Weight decay for the optimizer. Default is 1e-2.
      save_step (int): Step interval for saving the model checkpoints. Default is 10000.
      custom_model (str): Path to a custom model checkpoint (.pth file) to use instead of the default XTTS model.
      version (str): XTTS version to use (default: "main"). 
      max_text_length (int): Maximum text length for the model. Default is 200.
      max_audio_length (int): Maximum audio length for the model. Default is 255995 (approximately 12 seconds at 22050 Hz).
      multi_gpu (bool): Whether to use multi-GPU training. Default is False.
  Returns:
      tuple: Paths to the XTTS checkpoint, tokenizer file, config file, trainer output path, and speaker reference audio file.
  """
  RUN_NAME = "GPT_XTTS_FT"
  PROJECT_NAME = "XTTS_trainer_maltese"
  DASHBOARD_LOGGER = "tensorboard"
  LOGGER_URI = None
  num_workers = 8

  OUT_PATH = os.path.join(output_path, "run", "training") #Path.cwd()  #os.path.join(output_path, "run", "training")
  os.makedirs(OUT_PATH, exist_ok=True)

  # Training Parameters
  OPTIMIZER_WD_ONLY_ON_WEIGHTS = not multi_gpu
  START_WITH_EVAL = False #TODO
  BATCH_SIZE = batch_size
  GRAD_ACUMM_STEPS = grad_acumm

  print(f" > Training XTTS model for Maltese with {len(metadatas)} datasets, {num_epochs} epochs, batch size {BATCH_SIZE}, grad_acumm {GRAD_ACUMM_STEPS}, output path: {OUT_PATH}")
  print(" > Using the following datasets:")
  DATASETS_CONFIG_LIST = []
  for metadata in metadatas:
    train_csv, eval_csv, language = metadata.split(",")
    print(train_csv, eval_csv, language)
    if language == "ja":
      num_workers = 0

    config_dataset = BaseDatasetConfig(
      formatter="coqui",
      dataset_name="ft_dataset",
      path=os.path.dirname(train_csv), #TODO os.path.join(output_path, "dataset")
      meta_file_train=os.path.basename(train_csv),
      meta_file_val=os.path.basename(eval_csv),
      language=language,
    )
    DATASETS_CONFIG_LIST.append(config_dataset)

  print(" > Downloading XTTS model files...")
  MEL_NORM_FILE, DVAE_CHECKPOINT, XTTS_CHECKPOINT, TOKENIZER_FILE = download(output_path, version=version, custom_model=custom_model)
  print(" > XTTS model files downloaded successfully!")

  print("Setting up model arguments...")
  model_args = GPTArgs(
    max_conditioning_length=132300,  # 6 secs
    min_conditioning_length=66150,  # 3 secs   or 11025 for 0.5sec
    debug_loading_failures=False,
    max_wav_length=max_audio_length,
    max_text_length=max_text_length,
    mel_norm_file=MEL_NORM_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,
    tokenizer_file=TOKENIZER_FILE,
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True
  )

  audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

  config = GPTTrainerConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name=PROJECT_NAME,
    run_description="""GPT XTTS fine-tuning for Maltese""",
    dashboard_logger=DASHBOARD_LOGGER,
    logger_uri=LOGGER_URI, # type: ignore
    audio=audio_config,
    epochs=num_epochs,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=num_workers,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=100,
    save_step=save_step,
    save_n_checkpoints=1,
    save_checkpoints=True,
    # target_loss="loss",
    print_eval=False,
    optimizer="AdamW",
    optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": weight_decay},
    lr=lr,
    lr_scheduler="MultiStepLR", #TODO use CosineAnnealingLR?
    lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
    test_sentences=[],
  )

  model = GPTTrainer.init_from_config(config)

  print("Loading datasets...")
  train_samples, eval_samples = load_tts_samples(
    DATASETS_CONFIG_LIST,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
  )
  print(f" > Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples.")

  trainer = Trainer(
    TrainerArgs(
      restore_path=None, #type: ignore  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
      skip_train_epoch=False,
      start_with_eval=START_WITH_EVAL,
      grad_accum_steps=GRAD_ACUMM_STEPS,
    ),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
  )

  print("Starting training...")
  trainer.fit()
  print("Training finished!")

  print("Saving final model...")
  trainer.save_checkpoint(
    os.path.join(OUT_PATH, "final_model.pth"),
    config=config,
    tokenizer_file=TOKENIZER_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,
  )

  print("Saving configuration...")
  CONFIG_PATH = os.path.join(OUT_PATH, "config.json")
  inference_config = XttsConfig()
  inference_config.model_args = config.model_args  # Copy model args from training
  inference_config.audio = config.audio
  inference_config.save_json(CONFIG_PATH)
  print(f"Configuration saved to {CONFIG_PATH} and model checkpoint saved to {os.path.join(OUT_PATH, 'final_model.pth')}.")

  # get the longest text audio file to use as speaker reference
  samples_len = [len(item["text"].split(" ")) for item in train_samples] # type: ignore
  longest_text_idx =  samples_len.index(max(samples_len))
  speaker_ref = train_samples[longest_text_idx]["audio_file"] # type: ignore
  if not os.path.isabs(speaker_ref):
    speaker_ref = os.path.join(output_path, os.path.dirname(metadatas[0].split(",")[0]), speaker_ref)

  trainer_out_path = trainer.output_path

  # deallocate VRAM and RAM
  del model, trainer, train_samples, eval_samples
  gc.collect()

  return XTTS_CHECKPOINT, TOKENIZER_FILE, CONFIG_PATH, trainer_out_path, speaker_ref






# ========================== Inference ==========================
def inference(xtts_checkpoint, xtts_config, xtts_vocab, tts_text, speaker_audio_file, lang):
  """Run inference using the XTTS model with the provided configuration and text.
  Args:
      xtts_checkpoint (str): Path to the XTTS model checkpoint.
      xtts_config (str): Path to the XTTS configuration file.
      xtts_vocab (str): Path to the XTTS vocabulary file.
      tts_text (str): Text to be synthesized.
      speaker_audio_file (str): Path to the audio file of the speaker for conditioning.
      lang (str): Language code for the text. Supported languages include "en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko", and "mt".
  Returns:
      torch.Tensor: Synthesized audio waveform.
  """

  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  try:
    import deepspeed
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
  XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=use_deepspeed)
  XTTS_MODEL.to(device)
  print("Model loaded successfully!")


  gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, # type: ignore
    max_ref_length=XTTS_MODEL.config.max_ref_len, # type: ignore
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs, # type: ignore
  )

  import nltk
  from nltk.data import find

  try:
    find('tokenizers/punkt')
  except LookupError:
    print("NLTK 'punkt' tokenizer not found. downloading it now... (you can also download it manually using \"python -c \"import nltk; nltk.download('punkt')\"\")")
    nltk.download('punkt')
    print("NLTK 'punkt' tokenizer downloaded successfully.")
    pass

  from nltk.tokenize import sent_tokenize

  tts_texts = sent_tokenize(tts_text)

  wav_chunks = []
  print("Infering...")
  for text in tqdm(tts_texts):
    wav_chunk = XTTS_MODEL.inference(
      text=text,
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





# ========================== CLI Parser ==========================
def create_xtts_trainer_parser():
  """Create a command-line argument parser for the XTTS Trainer.
  """
  parser = argparse.ArgumentParser(description="Arguments for XTTS Trainer")
  parser.add_argument("--output_path", type=str, required=True, help="Path to pretrained + checkpoint model")
  parser.add_argument("--metadatas", nargs='+', type=str, required=True, help="train_csv_path,eval_csv_path,language")
  parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
  parser.add_argument("--batch_size", type=int, default=3, help="Mini batch size")
  parser.add_argument("--grad_acumm", type=int, default=84, help="Grad accumulation steps")
  parser.add_argument("--max_audio_length", type=int, default=255995, help="Max audio length")
  parser.add_argument("--max_text_length", type=int, default=200, help="Max text length")
  parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
  parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
  parser.add_argument("--save_step", type=int, default=10000, help="Save step")
  parser.add_argument("--custom_model", type=str, default="", help="Path to custom model checkpoint (.pth file)")
  parser.add_argument("--version", type=str, default="main", help="XTTS version to use (default: main)")
  parser.add_argument("--multi_gpu", action='store_true', help="Use multi-GPU training")
  parser.add_argument("--metadata_path", type=str, required=True, help="Path to a single metadata file for tokenizer training.")
  parser.add_argument("--language", type=str, required=True, help="Language code for the new language (e.g., 'mt').")
  parser.add_argument("--extended_vocab_size", type=int, default=2000, help="Vocabulary size for the new tokenizer.")
  # parser.add_argument("--no_deepspeed", action='store_true', help="Disable deepspeed for training")
  return parser


if __name__ == "__main__":
  parser = create_xtts_trainer_parser()
  args = parser.parse_args()

  # Step 1: Download the base XTTS model files.
  print("Step 1: Downloading XTTS base model files.")
  download(
    output_path=args.output_path,
    version=args.version
  )
  
  # Step 2: Extend the tokenizer for the new language.
  print("Step 2: Extending the XTTS tokenizer with the new language.")
  extend_tokenizer(
    output_path=args.output_path,
    metadata_path=args.metadata_path, #datasets/metadata_train.csv
    language=args.language,
    extended_vocab_size=args.extended_vocab_size #2000
  )
  
  # Step 3: Adjust the config file to include the new language.
  print("Step 3: Adjusting the config file.")
  adjust_config(
    output_path=args.output_path,
    version=args.version,
    language=args.language
  )

  # Step 4: Start the training process with the extended tokenizer and updated config.
  print("Step 4: Starting GPT training.")
  xtts_checkpoint, xtts_vocab, config, trainer_out_path, speaker_ref = train_gpt(
    metadatas=args.metadatas,
    output_path=args.output_path,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    grad_acumm=args.grad_acumm,
    weight_decay=args.weight_decay,
    lr=args.lr,
    max_text_length=args.max_text_length,
    max_audio_length=args.max_audio_length,
    save_step=args.save_step
  )
  
  print(f"Checkpoint saved in dir: {trainer_out_path}")
  print(f"Checkpoint saved in dir: {trainer_out_path}")


  run_inference = input("Do you want to run inference? (y/n): ").strip().lower()
  inference_text = input("Enter the text for inference (or leave empty to use default): ").strip() # Hija test tal-mudell tat-taħdit il-ġdid tiegħi, il-lingwa Maltija hija interessanti! Esperimenti u testijiet huma importanti biex niskopru l-possibbiltajiet tat-taħdit.
  if run_inference == 'y':
    print("Running inference...")
    audio = inference(
      xtts_checkpoint=xtts_checkpoint,
      xtts_config=config,
      xtts_vocab=xtts_vocab,
      tts_text=inference_text,
      speaker_audio_file=speaker_ref,
      lang=args.language
    )
    print("Inference completed!")
    torchaudio.save(os.path.join(trainer_out_path, "output_maltese.wav"), audio, 24000)

    try:
      from IPython.display import Audio
      Audio(audio, rate=24000) # Play audio (for Jupyter Notebook)
    except ImportError:
      print("IPython not available, audio playback not supported in this environment.")
  else:
    print("Skipping inference. You can run it later by calling the `inference()` function.")



# ========================== CLI ==========================
#Prepare the environment:
# python -m venv venv
# source venv/bin/activate  # On Windows: venv\Scripts\activate
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install git+https://github.com/coqui-ai/TTS.git@dev
# pip install -r requirements.txt


# Download NLTK sentence tokenizer data
# python -c "import nltk; nltk.download('punkt')"
# XTTS uses spacy for some languages. Even if not "mt",
# it's good practice to have the english model as a fallback.
# python -m spacy download en_core_web_sm




# CUDA_VISIBLE_DEVICES=0 python maltese_xtts.py \
# --output_path checkpoints/ \
# --metadatas datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,mt datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,mt \
# --num_epochs 100 \
# --batch_size 8 \
# --grad_acumm 84 \
# --max_text_length 200 \
# --max_audio_length 255995 \
# --weight_decay 1e-2 \
# --lr 5e-6 \
# --save_step 10000 \
# --custom_model "" \
# --version main \
# --metadata_path datasets/metadata_train.csv \
# --language mt \
# --extended_vocab_size 2000 \





# Phonetic transcription
# A rule-based script to transcribe Maltese text into IPA notation. An example is shown below.

# >> from masri.transcribe.g2p import text2phon
# >> print(text2phon("Ilbieraħ mort s'Għawdex"))
# ɪlbɪːrɐh mɔrt sɐʊdɛʃ
# Numbers to words
# An extension of num2words for the Maltese language. An example is shown below.

# >> from masri.transcribe.num2text import num2text
# >> print(num2text(301000))
# tliet mitt elf u  elf