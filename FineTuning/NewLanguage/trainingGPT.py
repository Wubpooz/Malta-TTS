import compatibility

import os
import gc
import torch
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets import load_tts_samples
from peft import get_peft_model, LoraConfig, TaskType

from dataclasses import dataclass
from enum import Enum

@dataclass
class XttsTrainingAudioConfig(XttsAudioConfig):
  """Extended XttsAudioConfig with dvae_sample_rate for training.
  """
  dvae_sample_rate: int = 22050

class ForgettingMitigation(str, Enum):
    NONE = "none"
    LORA = "LORA"
    FREEZE = "FREEZE"


#TODO
def freeze_base_model_layers(model, trainable_layers = None):
  """
  Freeze most layers, only train specific ones to prevent forgetting
  Arguments:
      model: The model to modify.
      trainable_layers: List of layer names to keep trainable.
  Returns:
      None
  Raises:
      ValueError: If the model is not a valid transformer model.
  """
  if trainable_layers is None:
    # Only train the last few layers and embedding
    trainable_layers = [
      "text_embedding",     # Allow text embedding to adapt
      "layers.29",          # Last transformer layer  
      "layers.28",          # Second to last layer
      "layers.27",          # Third to last layer
      "final_norm",         # Final normalization
      "lm_head"            # Language model head
    ]

  total_params = sum(p.numel() for p in model.parameters())
    # Freeze all parameters first
  for param in model.parameters():
    param.requires_grad = False
  
  # Unfreeze specific layers
  trainable_params = 0
  for name, param in model.named_parameters():
    for layer_name in trainable_layers:
      if layer_name in name:
        param.requires_grad = True
        trainable_params += param.numel()
        print(f"Unfrozen layer: {name} ({param.numel():,} params)")
        break
  
  print(f"Total params: {total_params:,}")
  print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
  print(f"Frozen params: {total_params-trainable_params:,} ({(total_params-trainable_params)/total_params*100:.1f}%)")



def train_gpt(metadatas: list[str], language: str, mel_norm_file: str, dvae_checkpoint: str, xtts_checkpoint: str, tokenizer_file: str, vocab_size: int, output_path: str, num_epochs: int = 100, batch_size: int = 3, grad_acumm: int = 84, lr: float = 5e-06, weight_decay: float = 1e-2, save_step: int = 10000, print_step: int = 200, max_text_length: int = 200, max_audio_length: int = 255995, multi_gpu: bool = False, optimizations: bool = False, tf32: bool = False, forgetting_mitigation: ForgettingMitigation = ForgettingMitigation.LORA):
  """Train the GPT XTTS model for Maltese language.
  This function sets up the training configuration, downloads necessary files, initializes the model, and starts the training process.
  It also saves the final model checkpoint and configuration files after training.
  Based on the XTTSv2 fine-tuning scripts.
  Arguments:
      metadatas (list[str]): A list of metadata strings in the format "train_csv_path,eval_csv_path,language".
      language (str): Language code for the training data.
      mel_norm_file (str): Path to the mel normalization file.
      dvae_checkpoint (str): Path to the DVAE checkpoint file.
      xtts_checkpoint (str): Path to the XTTS checkpoint file.
      tokenizer_file (str): Path to the tokenizer file.
      vocab_size (int): Vocabulary size for the tokenizer.
      output_path (str): Path to save the model checkpoints and outputs. Default is the current directory/"checkpoints".
      num_epochs (int): Number of epochs for training. Default is 100.
      batch_size (int): Mini batch size. Default is 3.
      grad_acumm (int): Gradient accumulation steps. Default is 84.
      lr (float): Learning rate for the optimizer. Default is 5e-6.
      weight_decay (float): Weight decay for the optimizer. Default is 1e-2.
      save_step (int): Step interval for saving the model checkpoints. Default is 10000.
      print_step (int): Step interval for printing training progress. Default is 200.
      max_text_length (int): Maximum text length for the model. Default is 200.
      max_audio_length (int): Maximum audio length for the model. Default is 255995 (approximately 12 seconds at 22050 Hz).
      multi_gpu (bool): Whether to use multi-GPU training. Default is False.
      optimizations (bool): Whether to apply optimizations for faster training. Default is False.
      tf32 (bool): Whether to use TensorFloat-32 (TF32) precision. Default is False.
  Returns:
      tuple: Paths to the XTTS checkpoint, tokenizer file, config file, trainer output path, and speaker reference audio file.
  Raises:
      FileNotFoundError: If any of the required files are not found.
  """
  RUN_NAME = "GPT_XTTS_FT"
  PROJECT_NAME = "XTTS_trainer_maltese"
  DASHBOARD_LOGGER = "tensorboard"
  LOGGER_URI = None
  cpu_count = os.cpu_count() or 1  # fallback to 1 if None
  num_workers = min(8, cpu_count - 1) if cpu_count > 1 else 1

  # Training Parameters
  OPTIMIZER_WD_ONLY_ON_WEIGHTS = True # whether to apply weight decay only on the output layer or also on bias and normalization layers
  START_WITH_EVAL = False
  BATCH_SIZE = batch_size
  GRAD_ACUMM_STEPS = grad_acumm


  if output_path is None:
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
  OUT_PATH = os.path.join(output_path, "training") #Path.cwd()
  os.makedirs(OUT_PATH, exist_ok=True)

  if not os.path.exists(xtts_checkpoint):
    raise FileNotFoundError(f"XTTS checkpoint not found at {xtts_checkpoint}")
  if not os.path.exists(tokenizer_file):
    raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_file}")
  if not os.path.exists(mel_norm_file):
    raise FileNotFoundError(f"Mel norm file not found at {mel_norm_file}")
  if not os.path.exists(dvae_checkpoint):
    raise FileNotFoundError(f"DVAE checkpoint not found at {dvae_checkpoint}")


  print(f" > Training XTTS model for Maltese with {len(metadatas)} datasets, {num_epochs} epochs, batch size {BATCH_SIZE}, grad_acumm {GRAD_ACUMM_STEPS}, output path: {OUT_PATH}")
  print(" > Using the following datasets:")
  DATASETS_CONFIG_LIST = []
  for metadata in metadatas:
    train_csv, eval_csv, language = metadata.split(",")
    print(train_csv, eval_csv, language)
    if not os.path.exists(train_csv):
      raise FileNotFoundError(f"Train CSV file not found: {train_csv}")
    if not os.path.exists(eval_csv):
      raise FileNotFoundError(f"Eval CSV file not found: {eval_csv}")

    if language == "ja":
      num_workers = 0

    # coqui format: "audio_file", "text", "speaker_name", "emotion_name"
    # ljspeech format: "audio_file", "text", "normalized_transcription"" audio_file|text|transcription|speaker_name
    config_dataset = BaseDatasetConfig(
      formatter="ljspeech",
      dataset_name="MASRI_HEADSET",
      path=os.path.dirname(train_csv),
      meta_file_train=os.path.basename(train_csv),
      meta_file_val=os.path.basename(eval_csv),
      language=language,
    )
    DATASETS_CONFIG_LIST.append(config_dataset)

  print("Setting up model arguments...")
  model_args = GPTArgs(
    max_conditioning_length=132300,  # 6 secs
    min_conditioning_length=66150,  # 3 secs   or 11025 for 0.5sec
    debug_loading_failures=True,
    max_wav_length=max_audio_length,
    max_text_length=max_text_length,
    mel_norm_file=mel_norm_file,
    xtts_checkpoint=xtts_checkpoint,
    gpt_checkpoint=None, #TODO ? "" by default
    dvae_checkpoint=dvae_checkpoint,
    tokenizer_file=tokenizer_file,
    gpt_num_audio_tokens=1026, #8194 default
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    # gpt_loss_text_ce_weight = 0.01,
    # gpt_loss_mel_ce_weight = 1.0,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
    gpt_number_text_tokens=vocab_size,
  
    # Replace None values with default values to fix initialization:
    clvp_checkpoint="",
    decoder_checkpoint=""
  )

  audio_config = XttsTrainingAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

  try:
    config = GPTTrainerConfig(
      output_path=OUT_PATH,
      model_args=model_args,
      run_name=RUN_NAME,
      project_name=PROJECT_NAME,
      run_description="""GPT XTTS fine-tuning for Maltese""",
      dashboard_logger=DASHBOARD_LOGGER,
      logger_uri=LOGGER_URI or "",
      audio=audio_config,
      epochs=num_epochs,
      batch_size=BATCH_SIZE,
      batch_group_size=48,
      eval_batch_size=BATCH_SIZE,
      num_loader_workers=num_workers,
      eval_split_max_size=256,
      print_step=print_step,
      plot_step=100,
      log_model_step=100,
      save_step=save_step,
      save_n_checkpoints=1,
      save_checkpoints=True,
      target_loss="", # "loss",
      print_eval=True,
      run_eval_steps=save_step,
      optimizer="AdamW",

      # Parameters optimization for adding a new language without catastrophic forgetting:
      optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
      optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": weight_decay},
      lr=lr,
      lr_scheduler="MultiStepLR",
      lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
      grad_clip=1.0,
      # Other LR scheduler option:
      # lr_scheduler = "CosineAnnealingWarmRestarts",
      # lr_scheduler_params={"T_0": save_step // 4, "T_mult": 1, "eta_min": lr * 0.01, "last_epoch": -1}, T_0 -> Restart every quarter of save_step, eta_min -> Minimum learning rate

      # Performance optimizations
      mixed_precision=optimizations,
      precision="fp16" if optimizations else "fp32",
      allow_tf32=tf32, # TensorFloat-32 tensor cores may be used in matrix multiplications on Ampere or newer GPUs. Default to False.
      # use_noise_augment=True ?

      # Replace None with default values to fix failure:
      test_sentences=[],
      model_dir="",
      phonemizer="",
      phoneme_language="",
      text_cleaner="",
      phoneme_cache_path="",
      characters="",
      loss_masking=False,
      wandb_entity=""
    )

    model = GPTTrainer.init_from_config(config)

    if forgetting_mitigation == ForgettingMitigation.FREEZE:
      print("Freezing base model layers...")
      freeze_base_model_layers(model.xtts.gpt)

    if forgetting_mitigation == ForgettingMitigation.LORA:
      print("Applying LoRA...")
      lora_config = LoraConfig(
        r=8,              # Rank of LoRA matrices
        lora_alpha=16,    # Scaling
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
      )
      model = get_peft_model(model, lora_config)

    print("Loading datasets...")
    train_samples, eval_samples = load_tts_samples(
      DATASETS_CONFIG_LIST,
      eval_split=True,
      eval_split_max_size=config.eval_split_max_size,
      eval_split_size=config.eval_split_size,
    )
    print(f" > Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples.")


    from utils import add_language_to_tokenizer
    add_language_to_tokenizer(model.xtts.tokenizer, lang_code=language)


    trainer = Trainer(
      TrainerArgs(
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

    # TODO untested
    if multi_gpu and torch.cuda.device_count() > 1:
      try:
        trainer.model = torch.nn.DataParallel(trainer.model, device_ids=list(range(torch.cuda.device_count())))
      except Exception as e:
        print(f"Error occurred while setting up multi-GPU training: {e}")


    print("Starting training...")
    trainer.fit()
    print("Training finished!")

    print("Saving final model...")
    trainer.save_checkpoint()

    print("Saving configuration...")
    CONFIG_PATH = os.path.join(OUT_PATH, "config.json")
    if os.path.exists(CONFIG_PATH):
      inference_config = XttsConfig()
      inference_config.model_args = config.model_args  # Copy model args from training
      inference_config.audio = config.audio
      inference_config.save_json(CONFIG_PATH)
      print(f"Configuration saved to {CONFIG_PATH} and model checkpoint saved to {os.path.join(OUT_PATH, 'final_model.pth')}.")
    else:
      print(f"Error: Configuration file not found at {CONFIG_PATH}. It was not created.")
      print(f"Parameters that would have been used are:")
      for key, value in config.model_args.items():
        print(f"  {key}: {value}")

    # Get the longest text audio file to use as speaker reference
    try:
      samples_len = [len(item["text"].split(" ")) for item in train_samples] # type: ignore
      longest_text_idx =  samples_len.index(max(samples_len))
      speaker_ref = train_samples[longest_text_idx]["audio_file"] # type: ignore
      if not os.path.isabs(speaker_ref) and output_path is not None and os.path.exists(speaker_ref):
        speaker_ref = os.path.join(output_path, os.path.dirname(metadatas[0].split(",")[0]), speaker_ref)
      print(f"Speaker reference: {speaker_ref}")
    except Exception as e:
      print(f"Error occurred while getting speaker reference: {e}")
      speaker_ref = "N/A"

    # Deallocate VRAM and RAM
    for var in ["model", "trainer", "train_samples", "eval_samples", "config"]:
      if var in locals():
        del locals()[var]
    torch.cuda.empty_cache()
    gc.collect()

    return xtts_checkpoint, tokenizer_file, CONFIG_PATH, trainer.output_path, speaker_ref

  except Exception as e:
    # Deallocate VRAM and RAM
    for var in ["model", "trainer", "train_samples", "eval_samples", "config"]:
      if var in locals():
        del locals()[var]
    torch.cuda.empty_cache()
    gc.collect()
    raise e




if __name__ == "__main__":
  from parsers import create_train_GPT_parser
  parser = create_train_GPT_parser()
  args = parser.parse_args()

  train_gpt(
    metadatas=args.metadatas,
    language=args.language,
    mel_norm_file=args.mel_norm_file,
    dvae_checkpoint=args.dvae_checkpoint,
    xtts_checkpoint=args.xtts_checkpoint,
    tokenizer_file=args.tokenizer_file,
    vocab_size=args.vocab_size,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    grad_acumm=args.grad_acumm,
    output_path=args.output_path,
    lr=args.lr,
    weight_decay=args.weight_decay,
    save_step=args.save_step,
    max_text_length=args.max_text_length,
    max_audio_length=args.max_audio_length,
    multi_gpu=args.multi_gpu,
    optimizations=args.optimizations,
    tf32=args.tf32,
    forgetting_mitigation=args.forgetting_mitigation
  )
  print("Training completed successfully!")
  print("You can now run inference using the trained model.")