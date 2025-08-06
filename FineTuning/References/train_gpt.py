# Based on https://github.com/daswer123/xtts-webui/blob/main/scripts/utils/gpt_train.py & https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages/blob/main/train_gpt_xtts.py

# ========================== CLI ==========================
# CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
# --output_path checkpoints/ \
# --metadatas datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,vi datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,vi \
# --num_epochs 5 \
# --batch_size 8 \
# --grad_acumm 4 \
# --max_text_length 400 \
# --max_audio_length 330750 \
# --weight_decay 1e-2 \
# --lr 5e-6 \
# --save_step 50000


# ========================== Code ==========================
import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize
from IPython.display import Audio

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def inference():
  xtts_checkpoint = "checkpoints/path/to/model.pth"
  xtts_config = "checkpoints/path/to/config.json"
  xtts_vocab = "checkpoints/path/to/vocab.json"

  tts_text = "Text"
  speaker_audio_file = "ref.wav"
  lang = "mt"

  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  config = XttsConfig()
  print("Loading config...")
  config.load_json(xtts_config)
  print("Config Loaded.")
  print("Initing model...")
  XTTS_MODEL = Xtts.init_from_config(config)
  print("Model Init, loadign checkpoint...")
  XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False) #True for acceleration
  XTTS_MODEL.to(device)
  print("Model loaded successfully!")


  gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, # type: ignore
    max_ref_length=XTTS_MODEL.config.max_ref_len, # type: ignore
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs, # type: ignore
  )

  tts_texts = sent_tokenize(tts_text)

  wav_chunks = []
  print("Infering...")
  for text in tqdm(tts_texts):
    wav_chunk = XTTS_MODEL.inference(
      text=text,
      language=lang,
      gpt_cond_latent=gpt_cond_latent,
      speaker_embedding=speaker_embedding,
      temperature=0.1,
      length_penalty=1.0,
      repetition_penalty=10.0,
      top_k=10,
      top_p=0.3,
    )
    wav_chunks.append(torch.tensor(wav_chunk["wav"]))
  print("Inference successful!")

  out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()
  Audio(out_wav, rate=24000) # Play audio (for Jupyter Notebook)





import os
import gc
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
import shutil

import argparse


def train_gpt(metadatas, num_epochs, batch_size, grad_acumm,output_path, lr=5e-06, weight_decay=1e-2, save_step=1000, custom_model="", version="main", max_text_length=200, max_audio_length=255995):
  RUN_NAME = "GPT_XTTS_FT"
  PROJECT_NAME = "XTTS_trainer"
  DASHBOARD_LOGGER = "tensorboard"
  LOGGER_URI = None  
  num_workers = 8

  # Set here the path that the checkpoints will be saved. Default: ./run/training/
  OUT_PATH = os.path.join(output_path, "run", "training")

  # Training Parameters
  OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
  START_WITH_EVAL = False
  BATCH_SIZE = batch_size
  GRAD_ACUMM_STEPS = grad_acumm

  DATASETS_CONFIG_LIST = []
  for metadata in metadatas:
    train_csv, eval_csv, language = metadata.split(",")
    print(train_csv, eval_csv, language)
    if language == "ja":
      num_workers = 0

    config_dataset = BaseDatasetConfig(
      formatter="coqui",
      dataset_name="ft_dataset",
      path=os.path.dirname(train_csv), #os.path.join(output_path, "dataset")
      meta_file_train=os.path.basename(train_csv),
      meta_file_val=os.path.basename(eval_csv),
      language=language,
    )

  DATASETS_CONFIG_LIST.append(config_dataset)


  CHECKPOINTS_OUT_PATH = os.path.join(Path.cwd(), "models",f"{version}")
  os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


  # DVAE files
  DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
  MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

  # Set the path to the downloaded files
  DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
  MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

  # download DVAE files if needed
  if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


  # Download XTTS v2.0 checkpoint if needed
  TOKENIZER_FILE_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/vocab.json"
  XTTS_CHECKPOINT_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/model.pth"
  XTTS_CONFIG_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/config.json"
  XTTS_SPEAKER_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/speakers_xtts.pth"

  # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
  TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
  XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file
  XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))  # config.json file
  XTTS_SPEAKER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_SPEAKER_LINK))  # speakers_xtts.pth file

  # download XTTS v2.0 files if needed
  if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(f" > Downloading XTTS v{version} files!")
    ModelManager._download_model_files(
      [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK, XTTS_CONFIG_LINK,XTTS_SPEAKER_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

  # Transfer this files to ready folder
  READY_MODEL_PATH = os.path.join(output_path,"ready")
  if not os.path.exists(READY_MODEL_PATH):
    os.makedirs(READY_MODEL_PATH)

  NEW_TOKENIZER_FILE = os.path.join(READY_MODEL_PATH, "vocab.json")
  # NEW_XTTS_CHECKPOINT = os.path.join(READY_MODEL_PATH, "model.pth")
  NEW_XTTS_CONFIG_FILE = os.path.join(READY_MODEL_PATH, "config.json")
  NEW_XTTS_SPEAKER_FILE = os.path.join(READY_MODEL_PATH, "speakers_xtts.pth")

  shutil.copy(TOKENIZER_FILE, NEW_TOKENIZER_FILE)
  # shutil.copy(XTTS_CHECKPOINT, os.path.join(READY_MODEL_PATH, "model.pth"))
  shutil.copy(XTTS_CONFIG_FILE, NEW_XTTS_CONFIG_FILE)
  shutil.copy(XTTS_SPEAKER_FILE, NEW_XTTS_SPEAKER_FILE)

  TOKENIZER_FILE = NEW_TOKENIZER_FILE # vocab.json file
  # XTTS_CHECKPOINT = NEW_XTTS_CHECKPOINT  # model.pth file
  XTTS_CONFIG_FILE = NEW_XTTS_CONFIG_FILE  # config.json file
  XTTS_SPEAKER_FILE = NEW_XTTS_SPEAKER_FILE  # speakers_xtts.pth file


  if custom_model != "":
    if os.path.exists(custom_model) and custom_model.endswith('.pth'):
      XTTS_CHECKPOINT = custom_model
      print(f" > Loading custom model: {XTTS_CHECKPOINT}")
    else:
      print(" > Error: The specified custom model is not a valid .pth file path.")


  model_args = GPTArgs(
    max_conditioning_length=132300,  # 6 secs
    min_conditioning_length=66150,  # 3 secs   or 11025 for 0.5sec
    debug_loading_failures=False,
    max_wav_length=max_audio_length,  # ~11.6 seconds
    max_text_length=max_text_length,
    mel_norm_file=MEL_NORM_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
    tokenizer_file=TOKENIZER_FILE,
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
  )


  audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

  # config = GPTTrainerConfig()
  # config.load_json(XTTS_CONFIG_FILE)

  config = GPTTrainerConfig(
    epochs=num_epochs,
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name=PROJECT_NAME,
    run_description="""
        GPT XTTS fine-tuning for Maltese
        """,
    dashboard_logger=DASHBOARD_LOGGER,
    logger_uri=LOGGER_URI, # type: ignore
    audio=audio_config,
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
    lr_scheduler="MultiStepLR",
    lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1}, # it was adjusted accordly for the new step scheme
    test_sentences=[],
  )


  model = GPTTrainer.init_from_config(config)

  train_samples, eval_samples = load_tts_samples(
    DATASETS_CONFIG_LIST,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
  )


  trainer = Trainer(
    TrainerArgs(
      restore_path="", # OR None  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
      skip_train_epoch=False,
      start_with_eval=START_WITH_EVAL,
      grad_accum_steps=GRAD_ACUMM_STEPS,
    ),
    config,
    output_path=OUT_PATH, #os.path.join(output_path, "run", "training")
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
  )
  trainer.fit()

  # get the longest text audio file to use as speaker reference
  samples_len = [len(item["text"].split(" ")) for item in train_samples] # type: ignore
  longest_text_idx =  samples_len.index(max(samples_len))
  speaker_ref = train_samples[longest_text_idx]["audio_file"] # type: ignore

  trainer_out_path = trainer.output_path

  # deallocate VRAM and RAM
  del model, trainer, train_samples, eval_samples
  gc.collect()

  return XTTS_SPEAKER_FILE,XTTS_CONFIG_FILE, XTTS_CHECKPOINT, TOKENIZER_FILE, trainer_out_path, speaker_ref


# For GUI
def create_xtts_trainer_parser():
  parser = argparse.ArgumentParser(description="Arguments for XTTS Trainer")
  parser.add_argument("--output_path", type=str, required=True, help="Path to pretrained + checkpoint model")
  parser.add_argument("--metadatas", nargs='+', type=str, required=True, help="train_csv_path,eval_csv_path,language")
  parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
  parser.add_argument("--batch_size", type=int, default=1, help="Mini batch size")
  parser.add_argument("--grad_acumm", type=int, default=1, help="Grad accumulation steps")
  parser.add_argument("--max_audio_length", type=int, default=255995, help="Max audio length")
  parser.add_argument("--max_text_length", type=int, default=200, help="Max text length")
  parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
  parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
  parser.add_argument("--save_step", type=int, default=5000, help="Save step")
  return parser


if __name__ == "__main__":
  parser = create_xtts_trainer_parser()
  args = parser.parse_args()

  trainer_out_path = train_gpt(
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