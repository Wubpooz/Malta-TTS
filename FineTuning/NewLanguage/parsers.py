import argparse

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


def create_inference_parser():
  """Create a command-line argument parser for XTTS Inference.
  """
  parser = argparse.ArgumentParser(description="Arguments for XTTS Inference")
  parser.add_argument("--model_path", type=str, required=True, help="Path to the trained XTTS model")
  parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated audio files")
  parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
  parser.add_argument("--language", type=str, required=True, help="Language code for the new language (e.g., 'mt').")
  return parser


def create_dataset_preparation_parser():
  """Create a command-line argument parser for dataset preparation.
  """
  parser = argparse.ArgumentParser(description="Arguments for Dataset Preparation")
  parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
  parser.add_argument("--output_path", type=str, required=True, help="Path to save the prepared dataset")
  parser.add_argument("--language", type=str, required=True, help="Language code for the new language (e.g., 'mt').")
  return parser


def create_tokenizer_training_parser():
  """Create a command-line argument parser for tokenizer training.
  """
  parser = argparse.ArgumentParser(description="Arguments for Tokenizer Training")
  parser.add_argument("--metadata_path", type=str, required=True, help="Path to the metadata file for tokenizer training")
  parser.add_argument("--output_path", type=str, required=True, help="Path to save the trained tokenizer")
  parser.add_argument("--language", type=str, required=True, help="Language code for the new language (e.g., 'mt').")
  parser.add_argument("--extended_vocab_size", type=int, default=2000, help="Vocabulary size for the new tokenizer.")
  return parser


def create_training_gpt_parser():
  """Create a command-line argument parser for GPT2 training.
  """
  parser = argparse.ArgumentParser(description="Arguments for GPT2 Training")
  parser.add_argument("--metadata_path", type=str, required=True, help="Path to the metadata file for tokenizer training")
  parser.add_argument("--output_path", type=str, required=True, help="Path to save the trained tokenizer")
  parser.add_argument("--language", type=str, required=True, help="Language code for the new language (e.g., 'mt').")
  parser.add_argument("--extended_vocab_size", type=int, default=2000, help="Vocabulary size for the new tokenizer.")
  return parser