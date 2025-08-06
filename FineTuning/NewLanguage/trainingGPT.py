import os
import gc
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets import load_tts_samples

from download import download

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