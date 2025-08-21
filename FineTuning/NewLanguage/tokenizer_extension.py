import compatibility

import gc
import os
import json
import torch
import shutil
import pandas as pd
from collections import Counter
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from masri.tokenise.tokenise import MTWordTokenizer
from utils import preprocess_maltese_text


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

    print(f"Checkpoint resized and saved to: {xtts_checkpoint_path}")
    print(f"Successfully resized from {current_vocab_size} to {new_vocab_size} tokens")
  
  else:
    print("No text embedding layer found in checkpoint")

  del checkpoint, old_embedding, new_embedding
  torch.cuda.empty_cache()
  gc.collect()
  
  return xtts_checkpoint_path



def extend_tokenizer(output_path, metadata_path, language, vocab_size=5_000, min_frequency=2, max_new_tokens=8_000):
  """
  Hybrid approach: Use MTWordTokenizer for linguistic preprocessing, then BPE for subword discovery.
  This combines Maltese linguistic awareness with BPE's ability to find optimal subword units.
  
  Arguments:
    - output_path: path to save the extended tokenizer
    - metadata_path: path to the metadata file
    - language: language code, e.g. 'mt'
    - vocab_size: size of the vocabulary for the new BPE tokenizer
    - min_frequency: minimum frequency for new tokens
    - max_new_tokens: maximum number of new tokens to add
  """
  print("==== Extending Tokenizer ====")
  # Load original tokenizer
  original_tokenizer_path = os.path.join(output_path, "vocab.json")
  if not os.path.exists(original_tokenizer_path):
    raise FileNotFoundError(f"vocab.json not found at {original_tokenizer_path}")

  original_tokenizer = Tokenizer.from_file(original_tokenizer_path)
  original_vocab = set(original_tokenizer.get_vocab().keys())

  # Load and preprocess texts
  df = pd.read_csv(metadata_path, sep="|", usecols=["text"])
  texts = df["text"].astype(str).tolist()
  if language == "mt":
    print("Applying Maltese-specific text preprocessing...")
    texts = [preprocess_maltese_text(text) for text in texts]


  print("Step 1: Creating linguistically-aware text chunks...")
  mt_tokenizer = MTWordTokenizer() if language == "mt" else None
  processed_texts = []
  for text in texts:
    if language == "mt" and mt_tokenizer:
      # Tokenize with MTWordTokenizer to get proper Maltese boundaries
      # Rejoin with spaces to create text that respects Maltese morphology
      mt_tokens = mt_tokenizer.tokenize(text)
      processed_text = " ".join(mt_tokens)
      processed_texts.append(processed_text)
    else:
      processed_texts.append(text)


  print("Step 2: Training BPE on linguistically-processed text...")
  new_tokenizer = Tokenizer(models.BPE())
  new_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() # type: ignore
  trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,  # type: ignore
    min_frequency=min_frequency, # type: ignore
    special_tokens=[]  # Don't let BPE trainer add special tokens # type: ignore
  )
  new_tokenizer.train_from_iterator(processed_texts, trainer)


  print("Step 3: Filtering BPE tokens for conflicts...")
  token_freq = Counter()
  for text in processed_texts:
    encoded = new_tokenizer.encode(text)
    for token in encoded.tokens:
      if token not in original_vocab:
        token_freq[token] += 1


  print("Step 4: Performing conflict detection...")
  safe_tokens = []

  for token, freq in token_freq.most_common():
    if freq < min_frequency:
      break
    if token in original_vocab:
      continue

    try:
      # Test if original tokenizer would break this token differently
      orig_encoding = original_tokenizer.encode(token)

      # If original tokenizer encodes it as a single existing token, skip
      if (len(orig_encoding.ids) == 1 and orig_encoding.tokens and orig_encoding.tokens[0] == token):
        continue

      # Additional safety check: ensure token doesn't contain problematic patterns
      if any(bad_char in token for bad_char in ['<', '>', '[', ']', '|']):
        continue

      # Check if it's a meaningful subword (not just punctuation or single chars)
      if len(token.strip()) < 2 and not token.isalnum():
        continue

      safe_tokens.append(token)
    except Exception as e:
      print(f"Warning: Skipping token '{token}' due to encoding error: {e}")
      continue


  print("Step 5: Adding missing character tokens...")
  all_chars = set()
  for text in texts:
    all_chars.update(list(text))
  missing_chars = [c for c in all_chars if original_tokenizer.token_to_id(c) is None and not c.isspace()]

  if missing_chars:
    original_tokenizer.add_tokens(missing_chars)
    print(f"Added {len(missing_chars)} missing character tokens.")


  print("Step 6: Finalizing new BPE tokens...")
  if len(safe_tokens) > max_new_tokens:
    safe_tokens = safe_tokens[:max_new_tokens]
    print(f"Limited to {max_new_tokens} new tokens (from {len(token_freq)} candidates)")

  if safe_tokens:
    original_tokenizer.add_tokens(safe_tokens)
    print(f"Added {len(safe_tokens)} new BPE-derived tokens.")
    print("Sample added tokens:", safe_tokens[:10])
  else:
    print("No safe BPE tokens found to add.")

  lang_tok = f"[{language}]"
  if original_tokenizer.token_to_id(lang_tok) is None:
    original_tokenizer.add_special_tokens([lang_tok])
    print(f"Added special token {lang_tok}")

  print("Step 8: Saving extended tokenizer...")
  shutil.copy2(original_tokenizer_path, os.path.join(output_path, "vocab_base.json"))
  tokenizer_json_path = os.path.join(output_path, "tokenizer.json")
  original_tokenizer.save(tokenizer_json_path)
  shutil.copy2(tokenizer_json_path, os.path.join(output_path, "vocab.json"))
  print(f"Extended tokenizer saved to {os.path.join(output_path, 'vocab.json')}.")


  print("Step 9: Updating model embeddings and config...")
  new_vocab_size = original_tokenizer.get_vocab_size()

  resize_xtts_checkpoint_embeddings(
    original_path=output_path,
    new_vocab_size=new_vocab_size
  )

  adjust_config(
    root=output_path,
    language=language,
    vocab_size=new_vocab_size
  )

  print(f"=== TOKENIZER EXTENSION COMPLETE ===")
  print(f"Final vocabulary size: {new_vocab_size}")
  print(f"Added {len(safe_tokens) + len(missing_chars)} new tokens total")

  del new_tokenizer, token_freq
  gc.collect()

  return new_vocab_size



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



def extend_tokenizer_with_validation(output_path, metadata_path, language, vocab_size=5_000, min_frequency=2, max_new_tokens=8_000):
  """
  Extended version with built-in validation to ensure no corruption occurred.
  """
  original_tokenizer_path = os.path.join(output_path, "vocab.json")
  validation_backup = os.path.join(output_path, "vocab_validation_backup.json")
  shutil.copy2(original_tokenizer_path, validation_backup)

  try:
    new_vocab_size = extend_tokenizer(output_path, metadata_path, language, vocab_size, min_frequency, max_new_tokens)

    print("\n=== RUNNING CORRUPTION VALIDATION ===")
    corruption_detected = debug_tokenizer_corruption(validation_backup, original_tokenizer_path)

    if corruption_detected:
      print("🔴 CORRUPTION DETECTED! Rolling back changes...")
      shutil.copy2(validation_backup, original_tokenizer_path)
      raise RuntimeError("Tokenizer extension failed validation - changes rolled back")
    else:
      print("✅ Validation passed - tokenizer extension successful!")
      os.remove(validation_backup)
      return new_vocab_size

  except Exception as e:
    if os.path.exists(validation_backup):
      shutil.copy2(validation_backup, original_tokenizer_path)
      os.remove(validation_backup)
    raise e



if __name__ == "__main__":
  from parsers import create_tokenizer_extension_parser
  parser = create_tokenizer_extension_parser()
  args = parser.parse_args()

  extend_tokenizer_with_validation(
    output_path=args.output_path,
    metadata_path=args.metadata_path,
    language=args.language,
    vocab_size=5000,
    min_frequency=2,
    max_new_tokens=8000
  )
