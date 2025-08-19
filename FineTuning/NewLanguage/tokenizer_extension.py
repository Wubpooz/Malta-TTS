import gc
import os
import json
import torch
import shutil
import pandas as pd
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace


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
  # backup_checkpoint_path = os.path.join(original_path, "model_backup.pth")
  # shutil.copyfile(xtts_checkpoint_path, backup_checkpoint_path)

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
    

    torch.save(checkpoint, xtts_checkpoint_path)
    # torch.save(checkpoint, xtts_checkpoint_path+".tmp")

    # if not os.path.exists(xtts_checkpoint_path):
    #   shutil.copy2(xtts_checkpoint_path+".tmp", xtts_checkpoint_path)

    print(f"Checkpoint resized and saved to: {xtts_checkpoint_path}")
    print(f"Successfully resized from {current_vocab_size} to {new_vocab_size} tokens")
  
  else:
    print("No text embedding layer found in checkpoint")

  del checkpoint, old_embedding, new_embedding
  torch.cuda.empty_cache()
  gc.collect()
  
  return xtts_checkpoint_path




def extend_tokenizer(output_path, metadata_path, language, min_frequency=2):
  """
  Extends the XTTS tokenizer with new vocabulary from the provided metadata file.
  Uses the Tokenizer API to preserve all config and special tokens. Avoid tokenID shifts which create scrambled audios.
  Arguments:
    - tokenizer_path: path to existing tokenizer JSON (tokenizer.json or vocab.json)
    - metadata_path: CSV with column 'text' (pipe-separated metadata used in the repo)
    - language: language code, e.g. 'mt'
    - min_frequency: only add tokens seen >= this threshold
  """
  tokenizer_json_path = os.path.join(output_path, "vocab.json")
  if not os.path.exists(tokenizer_json_path):
    raise FileNotFoundError(f"vocab.json not found at {tokenizer_json_path}")

  # load tokenizer
  tok = Tokenizer.from_file(tokenizer_json_path)
  tok.pre_tokenizer = Whitespace()

  # read texts
  df = pd.read_csv(metadata_path, sep="|", usecols=["text"])
  texts = df["text"].astype(str).tolist()

  # gather unique characters from the texts
  all_chars = set()
  for t in texts:
    all_chars.update(list(t))

  to_add_chars = [c for c in all_chars if tok.token_to_id(c) is None and not c.isspace()]

  if to_add_chars:
    tok.add_tokens(to_add_chars)
    print(f"Added {len(to_add_chars)} new character tokens.")


  # gather whitespace tokens + frequencies
  freq = Counter()
  for t in texts:
    for w in t.strip().split():
      w = w.strip()
      if not w:
        continue
      freq[w] += 1

  # decide which tokens to add
  to_add = []
  for token, count in freq.most_common():
    if count < min_frequency:
      break  # remaining tokens are below threshold (freq.most_common sorts descending)
    # exact existence check
    try:
      existing_id = tok.token_to_id(token)
    except Exception:
      existing_id = None
    if existing_id is not None:
      continue  # tokenizer already has the exact token

    # encode with original tokenizer to see representation
    enc = tok.encode(token)
    # if encoder returns a single token that matches the candidate textual token, treat as present
    if len(enc.ids) == 1 and enc.tokens and enc.tokens[0] == token:
      continue  # effectively present, skip adding

    # else this is a candidate new token; append
    to_add.append(token)

  # limit total additions (safety)
  MAX_ADD = 20000
  if len(to_add) > MAX_ADD:
    to_add = to_add[:MAX_ADD]

  if to_add:
    tok.add_tokens(to_add)
    print(f"Added {len(to_add)} new tokens (threshold={min_frequency}).")
  else:
    print("No new tokens to add by heuristic.")

  lang_tok = f"[{language}]"
  if tok.token_to_id(lang_tok) is None:
    tok.add_special_tokens([lang_tok])
    print(f"Added special token {lang_tok}")

  out_name = os.path.join(output_path, "tokenizer.json")
  tok.save(out_name)
  # many pipelines expect a 'vocab.json' or tokenizers accept tokenizer.json — keep a copy
  shutil.copy2(os.path.join(output_path, "vocab.json"), os.path.join(output_path, "vocab_base.json"))
  shutil.copy2(out_name, os.path.join(output_path, "vocab.json"))
  print(f"Tokenizer saved to {out_name} and copied to vocab.json")

  resize_xtts_checkpoint_embeddings(
    original_path=output_path,
    new_vocab_size=tok.get_vocab_size()
  )

  adjust_config(
    root=output_path,
    language=language,
    vocab_size=tok.get_vocab_size()
  )

  return tok.get_vocab_size()

# def extend_tokenizer_base(output_path: str, metadata_path: str, language: str, extended_vocab_size: int = 10_000):
#   """
#   Extends the XTTS tokenizer with new vocabulary from the provided metadata file.
#   Uses the Tokenizer API to preserve all config and special tokens.
#   """
#   root = os.path.join(output_path, "")
#   tokenizer_json_path = os.path.join(root, "vocab.json")
#   if not os.path.exists(tokenizer_json_path):
#     raise FileNotFoundError(f"vocab.json not found at {tokenizer_json_path}")

#   tokenizer = Tokenizer.from_file(tokenizer_json_path)
#   tokenizer.pre_tokenizer = Whitespace()

#   traindf = pd.read_csv(metadata_path, sep="|")
#   texts = traindf['text'].to_list()
#   trainer = BpeTrainer(vocab_size=extended_vocab_size, show_progress=True) # type: ignore
#   new_tokenizer = Tokenizer(BPE())
#   new_tokenizer.pre_tokenizer = Whitespace() # type: ignore
#   new_tokenizer.train_from_iterator(texts, trainer=trainer)

#   orig_vocab = tokenizer.get_vocab()
#   new_vocab = new_tokenizer.get_vocab()
#   missing_tokens = [tok for tok in new_vocab if tok not in orig_vocab]
#   if missing_tokens:
#     tokenizer.add_tokens(missing_tokens)
#     print(f"Added {len(missing_tokens)} new tokens.")

#   if f"[{language}]" not in orig_vocab:
#     tokenizer.add_special_tokens([f"[{language}]"])
#     print(f"Added special token: [{language}]")

#   tokenizer.save(tokenizer_json_path)
#   print(f"Extended tokenizer saved to {tokenizer_json_path}")

#   adjust_config(output_path, language, tokenizer.get_vocab_size())
#   resize_xtts_checkpoint_embeddings(output_path, tokenizer.get_vocab_size())
#   print("Vocabulary extension complete.")



def debug_tokenizer_corruption(original_tokenizer_path, extended_tokenizer_path):
  """Check if tokenizer extension corrupted existing token mappings"""

  print("=== TOKENIZER CORRUPTION DEBUG ===\n")
  print("Loading tokenizers...")
  try:
    original = Tokenizer.from_file(original_tokenizer_path)
    extended = Tokenizer.from_file(extended_tokenizer_path)
  except Exception as e:
    print(f"Error loading tokenizers: {e}")
    return False

  # Get vocabularies
  orig_vocab = original.get_vocab()
  ext_vocab = extended.get_vocab()

  print(f"Original vocab size: {len(orig_vocab)}")
  print(f"Extended vocab size: {len(ext_vocab)}")
  print()

  # Test common English words
  test_words = ["hello", "world", "the", "and", "is", "that", "you", "for", "magic", "doctor", "hi", "truth", "a", "two", "rt"]

  print("=== TOKEN ID COMPARISON ===")
  corruption_detected = False
  for word in test_words:
    orig_id = orig_vocab.get(word, "NOT_FOUND")
    ext_id = ext_vocab.get(word, "NOT_FOUND")

    if orig_id != ext_id:
      print(f"🔴 CORRUPTION: '{word}' changed from ID {orig_id} to {ext_id}")
      corruption_detected = True
    else:
      print(f"✅ OK: '{word}' kept ID {orig_id}")

  print()

  # Test tokenization of simple English text
  print("=== TOKENIZATION COMPARISON ===")
  test_text = "Hello world, how are you today?"

  orig_tokens = original.encode(test_text).ids
  ext_tokens = extended.encode(test_text).ids

  print(f"Original tokenization: {orig_tokens}")
  print(f"Extended tokenization:  {ext_tokens}")

  if orig_tokens != ext_tokens:
    print("🔴 CRITICAL: Same text produces different token sequences!")
    corruption_detected = True
  else:
    print("✅ OK: Same tokenization preserved")

  print()

  # Check for token ID shifts
  print("=== TOKEN ID SHIFT ANALYSIS ===")
  common_tokens = set(orig_vocab.keys()) & set(ext_vocab.keys())
  shifted_tokens = 0

  for token in list(common_tokens)[:50]:  # Check first 50 common tokens
    if orig_vocab[token] != ext_vocab[token]:
      shifted_tokens += 1

  shift_percentage = (shifted_tokens / min(50, len(common_tokens))) * 100
  print(f"Token ID shifts detected: {shifted_tokens}/50 ({shift_percentage:.1f}%)")

  if shift_percentage > 10:
    print("🔴 SEVERE: High percentage of token ID shifts detected!")
    corruption_detected = True

  print()
  print("=== DIAGNOSIS ===")
  if corruption_detected:
    print("🔴 TOKENIZER CORRUPTION CONFIRMED!")
  else:
    print("✅ Tokenizer appears intact - issue might be elsewhere")

  del original, extended
  gc.collect()



if __name__ == "__main__":
  from parsers import create_tokenizer_extension_parser
  parser = create_tokenizer_extension_parser()
  args = parser.parse_args()

  tokenizer = extend_tokenizer(
    output_path=args.output_path,
    metadata_path=args.metadata_path,
    language=args.language
  )

  # debug_tokenizer_corruption(os.path.join(args.output_path, "vocab_base.json"), os.path.join(args.output_path, "vocab.json"))
