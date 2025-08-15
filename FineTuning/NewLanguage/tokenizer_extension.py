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


# def add_language_to_VoiceBpeTokenizer(lang_code: str):
#   """Add a new language to the tokenizer.
#   Args:
#       lang (str): Language code for the new language to be added.
#   """
#   print(f"Adding new language: {lang_code}")
#   import TTS.tts.layers.xtts.tokenizer as tokenizer
#   import re

#   _original_preprocess_text = tokenizer.VoiceBpeTokenizer.preprocess_text

#   def custom_preprocess_text(self, txt, lang):
#     if lang == lang_code:
#       txt = txt.lower()
#       txt = re.sub(re.compile(r"\s+"), " ", txt)
#       # transliterate ?
#       return txt.strip()
#     return _original_preprocess_text(self, txt, lang)

#   # Monkey-patch
#   tokenizer.VoiceBpeTokenizer.preprocess_text = custom_preprocess_text




# def _merge_vocabs_preserve_ids(old_tokenizer_path, new_tokenizer_path, output_path):
#   """
#   Merges two vocabularies, preserving the token IDs from the old tokenizer
#   and adding new tokens from the new tokenizer.
#   """
#   print(f"Merging tokenizers from {old_tokenizer_path} and {new_tokenizer_path} into {output_path}")
#   with open(os.path.join(old_tokenizer_path, 'vocab.json'), encoding='utf-8') as f:
#     old_tokenizer = json.load(f)
#   with open(os.path.join(new_tokenizer_path, 'vocab.json'), encoding='utf-8') as f:
#     new_tokenizer = json.load(f)

#   old_vocab = old_tokenizer
#   new_vocab = new_tokenizer
#   print(f"Old tokenizer vocabulary size: {len(old_vocab)}")
#   print(f"New tokenizer vocabulary size: {len(new_vocab)}")

#   combined_vocab = old_vocab.copy()
#   new_id = max(old_vocab.values())
  
#   # Iterate through the new vocabulary and add words not in the old one
#   new_tokens = set(new_vocab.keys()) - set(old_vocab.keys())
#   print(f"{len(new_tokens)} New tokens: {list(new_tokens)[:20]}")

#   for token in new_tokens:
#     new_id += 1
#     combined_vocab[token] = new_id
#   print(f"Combined vocabulary size: {len(combined_vocab)}")

#   merged_tokenizer = old_tokenizer.copy()
#   merged_tokenizer = combined_vocab

#   os.makedirs(output_path, exist_ok=True)
#   with open(os.path.join(output_path, 'vocab.json'), 'w', encoding='utf-8') as fp:
#     json.dump(merged_tokenizer, fp, ensure_ascii=False, indent=2)
#   print(f"Combined vocabulary saved to {os.path.join(output_path, 'vocab.json')}")
#   return merged_tokenizer


# def extend_tokenizer(output_path: str, metadata_path: str, language: str, extended_vocab_size: int = 10_000):
#   """Extends the XTTS tokenizer with new vocabulary from the provided metadata file.
#   This function combines the existing tokenizer with a new tokenizer trained on the provided metadata.
#   It saves the new tokenizer in a specified directory and updates the vocabulary to include new tokens.
#   Args:
#       output_path (str): Path to the output directory where the tokenizer files will be saved.
#       metadata_path (str): Path to the metadata file containing training data.
#       language (str): Language code for the new language to be added.
#       extended_vocab_size (int): Desired size of the extended vocabulary. Default is 10_000.
#   """
#   root = os.path.join(output_path, "")
#   old_tokenizer_path = os.path.join(root, "old_tokenizer/")
#   new_tokenizer_path = os.path.join(root, "new_tokenizer/")
#   merged_tokenizer_path = os.path.join(root, "merged_tokenizer/")

#   if not os.path.exists(root):
#     raise FileNotFoundError(f"Output directory not found at {root}. Please ensure the path is correct.")
#   if not os.path.exists(metadata_path):
#     raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please ensure the path is correct.")

#   existing_tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
#   print(f"Original tokenizer loaded with {len(existing_tokenizer.get_vocab())} tokens.")

#   traindf = pd.read_csv(metadata_path, sep="|")
#   texts = traindf['text'].to_list()
#   new_tokenizer = Tokenizer(BPE())
#   new_tokenizer.pre_tokenizer = Whitespace() # type: ignore
#   trainer = BpeTrainer(special_tokens=[f"[{language}]"], vocab_size=extended_vocab_size) # type: ignore
#   print(f"Training new tokenizer with {len(texts)} texts...")
#   new_tokenizer.train_from_iterator(iter(texts), trainer=trainer)
#   new_tokenizer.add_special_tokens([f"[{language}]"])
#   print(f"New tokenizer trained with {len(new_tokenizer.get_vocab())} tokens.")

#   # Saving tokenizers
#   os.makedirs(new_tokenizer_path, exist_ok=True)
#   os.makedirs(old_tokenizer_path, exist_ok=True)
#   existing_tokenizer.model.save(old_tokenizer_path)
#   new_tokenizer.model.save(new_tokenizer_path)

#   print(f"Making backups at {os.path.join(root, 'backups/')}...")
#   os.makedirs(os.path.join(root, "backups/"), exist_ok=True)
#   shutil.copy2(os.path.join(root, "vocab.json"), os.path.join(root, "backups/vocab.json"))
#   shutil.copy2(os.path.join(root, "config.json"), os.path.join(root, "backups/config.json"))


#   _merge_vocabs_preserve_ids(old_tokenizer_path, new_tokenizer_path, merged_tokenizer_path)
#   # shutil.copy2(os.path.join(new_tokenizer_path, "merges.txt"), os.path.join(merged_tokenizer_path, "merges.txt"))
#   merged_vocab_file = os.path.join(merged_tokenizer_path, 'vocab.json')
#   new_merges_file = os.path.join(merged_tokenizer_path, 'merges.txt')

#   final_tokenizer = Tokenizer(BPE.from_file(cls=BPE, vocab=merged_vocab_file, merges=new_merges_file)) # type: ignore
#   final_tokenizer.pre_tokenizer = Whitespace() # type: ignore
#   special_tokens = [f"[{language}]", existing_tokenizer.get_vocab()[f"[{language}]"]]
#   final_tokenizer.add_special_tokens(special_tokens)
#   final_tokenizer.save(os.path.join(root, "vocab.json"))
#   final_tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
#   print(f"Final tokenizer loaded with {len(final_tokenizer.get_vocab())} tokens.")

#   print("Cleaning up...")
#   shutil.rmtree(old_tokenizer_path)
#   shutil.rmtree(new_tokenizer_path)
#   shutil.rmtree(merged_tokenizer_path)

#   print(f"Tokenizer has been successfully extended and saved to {os.path.join(root, 'vocab.json')}")

#   print("Updating the XTTS config file...")
#   adjust_config(output_path, language, len(final_tokenizer.get_vocab()))

#   print("Resizing the XTTS checkpoint...")
#   resize_xtts_checkpoint_embeddings(output_path, len(final_tokenizer.get_vocab()))

#   print("Vocabulary extension complete.")

def extend_tokenizer(output_path: str, metadata_path: str, language: str, extended_vocab_size: int = 10_000):
  """
  Extends the XTTS tokenizer with new vocabulary from the provided metadata file.
  Uses the Tokenizer API to preserve all config and special tokens.
  """
  import pandas as pd
  from tokenizers import Tokenizer
  from tokenizers.models import BPE
  from tokenizers.pre_tokenizers import Whitespace
  from tokenizers.trainers import BpeTrainer

  root = os.path.join(output_path, "")
  tokenizer_json_path = os.path.join(root, "vocab.json")
  if not os.path.exists(tokenizer_json_path):
    raise FileNotFoundError(f"vocab.json not found at {tokenizer_json_path}")

  tokenizer = Tokenizer.from_file(tokenizer_json_path)
  tokenizer.pre_tokenizer = Whitespace()

  traindf = pd.read_csv(metadata_path, sep="|")
  texts = traindf['text'].to_list()
  trainer = BpeTrainer(vocab_size=extended_vocab_size, show_progress=True)
  new_tokenizer = Tokenizer(BPE())
  new_tokenizer.pre_tokenizer = Whitespace()
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