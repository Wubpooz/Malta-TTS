import argparse

def create_xtts_trainer_parser():
  """Create a command-line argument parser for the XTTS Trainer.
  """
  parser = argparse.ArgumentParser(description="Arguments for XTTS Trainer")
  parser.add_argument("--is_download", action='store_true', help="Flag to indicate if the XTTS model files should be downloaded.")
  parser.add_argument("--is_tokenizer_extension", action='store_true', help="Flag to indicate if the tokenizer should be extended with a new language.")
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
  parser.add_argument("--mel_norm_file", type=str, required=False, help="Path to the mel normalization file")
  parser.add_argument("--dvae_checkpoint", type=str, required=False, help="Path to the pretrained DVAE model checkpoint (.pth file)")
  parser.add_argument("--xtts_checkpoint", type=str, required=False, help="Path to the pretrained XTTS model checkpoint (.pth file)")
  parser.add_argument("--tokenizer_file", type=str, required=False, help="Path to the tokenizer file (.json file)")
  parser.add_argument("--optimizations", action='store_true', help="Enable optimizations for training")
  parser.add_argument("--tf32", action='store_true', help="Enable TF32 for training")
  parser.add_argument("--forgetting_mitigation", type=str, choices=["none", "LORA", "FREEZE"], default="LORA", help="Method to mitigate forgetting during training (default: LORA)")
  # parser.add_argument("--no_deepspeed", action='store_true', help="Disable deepspeed for training")
  return parser


def create_download_parser():
  """Create a command-line argument parser for the XTTS model download script.
  """
  parser = argparse.ArgumentParser(description="Download XTTS model files")
  parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory")
  parser.add_argument("--version", type=str, default="main", help="Version of the XTTS model to download (default: main)")
  parser.add_argument("--custom_model", type=str, default="", help="Path to a custom model checkpoint (.pth file) to use instead of the default XTTS model")
  return parser


def create_inference_parser():
  """Create a command-line argument parser for the XTTS inference script.
  """
  parser = argparse.ArgumentParser(description="Arguments for XTTS Inference")
  parser.add_argument("--xtts_checkpoint", type=str, required=True, help="Path to the XTTS model checkpoint (.pth file)")
  parser.add_argument("--xtts_config", type=str, required=True, help="Path to the XTTS model config file (config.json)")
  parser.add_argument("--xtts_vocab", type=str, required=True, help="Path to the XTTS model vocabulary file (vocab.json)")
  parser.add_argument("--tts_text", type=str, required=True, help="Text to synthesize")
  parser.add_argument("--speaker_audio_file", type=str, required=True, help="Path to the speaker audio file for conditioning")
  parser.add_argument("--lang", type=str, default="en", help="Language code for the text (default: 'en')")
  parser.add_argument("--output_file", type=str, default=None, help="Output audio file path (default: 'output.wav')")
  parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling (default: 0.7)")
  parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty for sampling (default: 1.0)")
  parser.add_argument("--repetition_penalty", type=float, default=10.0, help="Repetition penalty for sampling (default: 10.0)")
  parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling (default: 50)")
  parser.add_argument("--top_p", type=float, default=0.8, help="Top-p for sampling (default: 0.8)")
  parser.add_argument("--LORA_trained", action='store_true', help="Whether the model was trained with LORA")
  return parser


def create_prepare_dataset_parser():
  """Create a command-line argument parser for the dataset preparation script.
  """
  parser = argparse.ArgumentParser(description="Prepare MASRI-HEADSET CORPUS v2 for Hugging Face.")
  parser.add_argument("--input_dir", type=str, required=True, help="Path to the root of the MASRI-HEADSET CORPUS v2 dataset.")
  parser.add_argument("--output_dir", type=str, required=True, help="Path where the prepared dataset will be saved.")
  parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of the dataset to include in the test split (default: 0.1).")
  parser.add_argument("--upload_to_hf", action='store_true', help="If set, will upload the dataset to Hugging Face after preparation.")
  parser.add_argument("--dataset_name", type=str, default="Bluefir/maltese-headset-v2", help="Name of the dataset to upload to Hugging Face (default: 'Bluefir/maltese-headset-v2').")
  return parser


def create_tokenizer_extension_parser():
  """Create a command-line argument parser for the tokenizer extension script.
  """
  parser = argparse.ArgumentParser(description="Extend the XTTS tokenizer with new vocabulary.")
  parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory where the tokenizer files will be saved.")
  parser.add_argument("--metadata_path", type=str, required=True, help="Path to the metadata file containing training data.")
  parser.add_argument("--language", type=str, required=True, help="Language code for the new language to be added.")
  return parser


def create_train_GPT_parser():
  """Create a command-line argument parser for the GPT training script.
  """
  parser = argparse.ArgumentParser(description="Train GPT model for XTTS.")
  parser.add_argument("--metadatas", nargs='+', type=str, required=True, help="List of metadata strings in the format 'train_csv_path,eval_csv_path,language'.")
  parser.add_argument("--language", type=str, default="mt", help="Language code for the new language (default: 'mt').")
  parser.add_argument("--mel_norm_file", type=str, required=True, help="Path to the mel normalization file.")
  parser.add_argument("--dvae_checkpoint", type=str, required=True, help="Path to the pretrained DVAE model checkpoint (.pth file).")
  parser.add_argument("--xtts_checkpoint", type=str, required=True, help="Path to the pretrained XTTS model checkpoint (.pth file).")
  parser.add_argument("--tokenizer_file", type=str, required=True, help="Path to the tokenizer file (.json file).")
  parser.add_argument("--vocab_size", type=int, default=6681, help="Vocabulary size for the tokenizer (default: 6681).")
  parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training (default: 10).")
  parser.add_argument("--batch_size", type=int, default=3, help="Batch size for training (default: 3).")
  parser.add_argument("--grad_acumm", type=int, default=84, help="Gradient accumulation steps (default: 84).")
  parser.add_argument("--max_audio_length", type=int, default=255995, help="Maximum audio length in samples (default: 255995).")
  parser.add_argument("--max_text_length", type=int, default=200, help="Maximum text length (default: 200).")
  parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for the optimizer (default: 1e-2).")
  parser.add_argument("--output_path", type=str, required=True, help="Path to save the model checkpoints and outputs.")
  parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for the optimizer (default: 5e-6).")
  parser.add_argument("--save_step", type=int, default=10000, help="Step interval for saving model checkpoints (default: 10000).")
  parser.add_argument("--multi_gpu", action='store_true', help="Whether to use multi-GPU training.")
  parser.add_argument("--optimizations", action='store_true', help="Whether to enable optimizations for training.")
  parser.add_argument("--tf32", action='store_true', help="Whether to enable TF32 for training.")
  parser.add_argument("--forgetting_mitigation", type=str, choices=["none", "LORA", "FREEZE"], default="LORA", help="Method to mitigate forgetting during training (default: LORA).")
  return parser


def create_train_DVAE_parser():
  """Create a command-line argument parser for the DVAE training script.
  """
  parser = argparse.ArgumentParser(description="Train Discrete VAE (DVAE) for XTTS.")
  parser.add_argument("--dvae_pretrained", type=str, required=True, help="Path to the pretrained DVAE model checkpoint (.pth file).")
  parser.add_argument("--mel_norm_file", type=str, required=True, help="Path to the mel normalization file.")
  parser.add_argument("--metadatas", nargs='+', type=str, required=True, help="List of metadata strings in the format 'train_csv_path,eval_csv_path,language'.")
  parser.add_argument("--language", type=str, default="mt", help="Language code for the new language (default: 'mt').")
  parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for the optimizer (default: 5e-6).")
  parser.add_argument("--grad_clip_norm", type=float, default=0.5, help="Gradient clipping norm (default: 0.5).")
  parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training (default: 5).")
  parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training (default: 512).")
  
  return parser