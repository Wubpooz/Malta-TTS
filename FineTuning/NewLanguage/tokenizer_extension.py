import os
import json
import torch
import shutil
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def adjust_config(root: str, language: str, vocab_size: int):
  """Adjust the XTTS configuration file to include the new language.
  Args:
      root (str): Path to the output directory where the config file is located. 
      language (str): Language code for the new language to be added.
      vocab_size (int): Desired size of the extended vocabulary.
  """
  config_path = os.path.join(root, "config.json")
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at {config_path}. Please ensure the path is correct.")
  with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
  
  if language not in config["languages"]:
    config["languages"].append(language)
  config["model_args"]["gpt_number_text_tokens"] = vocab_size

  with open(config_path, 'w', encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
  print(f"Updated config file saved to {config_path}. Added new language: {language}. Vocab size: {vocab_size}")


def resize_xtts_checkpoint_embeddings(original_path: str, new_vocab_size: int):
  """Resizes embedding layers to match new vocabulary size while preserving existing weights.
  Args:
      original_path (str): Path to the original XTTS checkpoint directory.
      new_vocab_size (int): New vocabulary size after tokenizer extension.
  """
  xtts_checkpoint_path = os.path.join(original_path, "model.pth")
  if not os.path.exists(xtts_checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found at {xtts_checkpoint_path}")
  
  print(f"Backing up checkpoint: {xtts_checkpoint_path + ' -> model_backup.pth'}")
  backup_checkpoint_path = os.path.join(original_path, "model_backup.pth")
  shutil.copyfile(xtts_checkpoint_path, backup_checkpoint_path)

  print(f"Resizing checkpoint embeddings: {xtts_checkpoint_path}")
  
  checkpoint = torch.load(xtts_checkpoint_path, map_location="cpu")

  if "gpt.text_embedding.weight" in checkpoint["model"]:
    current_vocab_size = checkpoint["model"]["gpt.text_embedding.weight"].shape[0]
    embedding_dim = checkpoint["model"]["gpt.text_embedding.weight"].shape[1]
    print(f"Current vocab size: {current_vocab_size}, New vocab size: {new_vocab_size}")
    
    if current_vocab_size == new_vocab_size:
      print("Vocabulary sizes match, no resizing needed.")
      return xtts_checkpoint_path
    
    old_embedding = checkpoint["model"]["gpt.text_embedding.weight"]
    new_embedding = torch.zeros(new_vocab_size, embedding_dim, dtype=old_embedding.dtype)
    
    # Copy existing embeddings
    min_vocab_size = min(current_vocab_size, new_vocab_size)
    new_embedding[:min_vocab_size] = old_embedding[:min_vocab_size]
    
    # Initialize new embeddings with small random values (similar to original initialization)
    if new_vocab_size > current_vocab_size:
      std = old_embedding.std().item()
      new_embedding[current_vocab_size:].normal_(mean=0.0, std=std)
      print(f"Initialized {new_vocab_size - current_vocab_size} new embeddings with std={std:.6f}")
    
    checkpoint["model"]["gpt.text_embedding.weight"] = new_embedding
    
    # Resize text_head.weight (output layer)
    if "gpt.text_head.weight" in checkpoint["model"]:
      old_head_weight = checkpoint["model"]["gpt.text_head.weight"]
      new_head_weight = torch.zeros(new_vocab_size, embedding_dim, dtype=old_head_weight.dtype)
      new_head_weight[:min_vocab_size] = old_head_weight[:min_vocab_size]
      
      if new_vocab_size > current_vocab_size:
        std = old_head_weight.std().item()
        new_head_weight[current_vocab_size:].normal_(mean=0.0, std=std)
      
      checkpoint["model"]["gpt.text_head.weight"] = new_head_weight
    
    # Resize text_head.bias
    if "gpt.text_head.bias" in checkpoint["model"]:
      old_bias = checkpoint["model"]["gpt.text_head.bias"]
      new_bias = torch.zeros(new_vocab_size, dtype=old_bias.dtype)
      new_bias[:min_vocab_size] = old_bias[:min_vocab_size]
      # New bias entries remain zero (good default)
      checkpoint["model"]["gpt.text_head.bias"] = new_bias
    

    backup_path = xtts_checkpoint_path + ".backup"
    if os.path.exists(xtts_checkpoint_path):
      shutil.copy2(xtts_checkpoint_path, backup_path)
      print(f"Backup created at: {backup_path}")
    
    torch.save(checkpoint, xtts_checkpoint_path)
    print(f"Checkpoint resized and saved to: {xtts_checkpoint_path}")
    print(f"Successfully resized from {current_vocab_size} to {new_vocab_size} tokens")
  
  else:
    print("No text embedding layer found in checkpoint")
  
  return xtts_checkpoint_path


def extend_tokenizer(output_path: str, metadata_path: str, language: str, extended_vocab_size: int = 10_000):
  """
  Extends the XTTS tokenizer with new vocabulary from the provided metadata file.
  Uses the Tokenizer API to preserve all config and special tokens.
  """
  root = os.path.join(output_path, "")
  tokenizer_json_path = os.path.join(root, "vocab.json")
  if not os.path.exists(tokenizer_json_path):
    raise FileNotFoundError(f"vocab.json not found at {tokenizer_json_path}")

  tokenizer = Tokenizer.from_file(tokenizer_json_path)
  tokenizer.pre_tokenizer = Whitespace()

  traindf = pd.read_csv(metadata_path, sep="|")
  texts = traindf['text'].to_list()
  trainer = BpeTrainer(vocab_size=extended_vocab_size, show_progress=True) # type: ignore
  new_tokenizer = Tokenizer(BPE())
  new_tokenizer.pre_tokenizer = Whitespace() # type: ignore
  new_tokenizer.train_from_iterator(texts, trainer=trainer)

  orig_vocab = tokenizer.get_vocab()
  new_vocab = new_tokenizer.get_vocab()
  missing_tokens = [tok for tok in new_vocab if tok not in orig_vocab]
  if missing_tokens:
    tokenizer.add_tokens(missing_tokens)
    print(f"Added {len(missing_tokens)} new tokens.")

  if f"[{language}]" not in orig_vocab:
    tokenizer.add_special_tokens([f"[{language}]"])
    print(f"Added special token: [{language}]")

  tokenizer.save(tokenizer_json_path)
  print(f"Extended tokenizer saved to {tokenizer_json_path}")

  adjust_config(output_path, language, tokenizer.get_vocab_size())
  resize_xtts_checkpoint_embeddings(output_path, tokenizer.get_vocab_size())
  print("Vocabulary extension complete.")


if __name__ == "__main__":
  from parsers import create_tokenizer_extension_parser
  parser = create_tokenizer_extension_parser()
  args = parser.parse_args()

  extend_tokenizer(
    output_path=args.output_path,
    metadata_path=args.metadata_path,
    language=args.language,
    extended_vocab_size=args.extended_vocab_size
  )