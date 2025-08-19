import gc
import os
import json
import torch
import shutil
import pandas as pd
from tokenizers import Tokenizer, AddedToken
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
    
    torch.save(checkpoint, xtts_checkpoint_path)
    print(f"Checkpoint resized and saved to: {xtts_checkpoint_path}")
    print(f"Successfully resized from {current_vocab_size} to {new_vocab_size} tokens")
  
  else:
    print("No text embedding layer found in checkpoint")

  del checkpoint, old_embedding, new_embedding
  torch.cuda.empty_cache()
  gc.collect()
  
  return xtts_checkpoint_path





# ==============================================================================================
# ======================================== Simple Fix ? ========================================
# ==============================================================================================
def extend_tokenizer_fix(output_path: str, metadata_path: str, language: str):
  tokenizer = Tokenizer.from_file(os.path.join(output_path, "vocab.json"))
  tokenizer.pre_tokenizer = Whitespace()

  print(f"Reading new texts from {metadata_path}")
  traindf = pd.read_csv(metadata_path, sep="|")
  texts = traindf['text'].to_list()

  # canonicalize your new tokens
  candidate_tokens = set()
  for text in texts:
    for tok in text.strip().split():
      tok = tok.strip()
      if tok and " " not in tok:
        candidate_tokens.add(tok)

  existing = tokenizer.get_vocab()
  to_add = [t for t in candidate_tokens if t not in existing]

  specials = []
  if f"[{language}]" not in existing:
    specials.append(AddedToken(f"[{language}]", single_word=True, special=True))

  if to_add:
    n_added = tokenizer.add_tokens(to_add)
    print("added tokens:", n_added)
  if specials:
    n_spec = tokenizer.add_special_tokens(specials)
    print("added specials:", n_spec)

  new_vocab = tokenizer.get_vocab()
  for k, v in existing.items():
    if new_vocab.get(k) != v:
      raise RuntimeError(f"ID shift detected for token: {k} {v} -> {new_vocab.get(k)}")

  tokenizer.save(os.path.join(output_path, "vocab.json") + ".new")





# =================================================================================================
# ======================================== Claude Solution ========================================
# =================================================================================================
def extend_tokenizer_BPE_training(output_path: str, metadata_path: str, language: str, extended_vocab_size: int = 10_000):
  """
  Safely extends the XTTS tokenizer by adding new tokens from the provided metadata file.
  Preserves existing token IDs and BPE merges while adding only genuinely new tokens.
  """
  import pandas as pd
  from tokenizers import Tokenizer
  from tokenizers.pre_tokenizers import Whitespace
  from tokenizers.models import BPE
  from tokenizers.trainers import BpeTrainer
  from tqdm import tqdm
  import tempfile

  root = os.path.join(output_path, "")
  tokenizer_json_path = os.path.join(root, "vocab.json")
  if not os.path.exists(tokenizer_json_path):
    raise FileNotFoundError(f"vocab.json not found at {tokenizer_json_path}")

  print(f"Loading existing tokenizer from {tokenizer_json_path}")
  existing_tokenizer = Tokenizer.from_file(tokenizer_json_path)
  existing_vocab = existing_tokenizer.get_vocab()
  
  # Save existing tokenizer state
  with tempfile.TemporaryDirectory() as tmp_dir:
      existing_vocab_path = os.path.join(tmp_dir, "existing_vocab.json")
      existing_merges_path = os.path.join(tmp_dir, "existing_merges.txt")
      existing_tokenizer.save(existing_vocab_path)
      
      # Extract existing BPE merges
      print("Extracting existing BPE merges...")
      model_path = os.path.join(root, "merges.txt")
      if os.path.exists(model_path):
          with open(model_path, 'r', encoding='utf-8') as f:
              existing_merges = f.readlines()
      else:
          existing_merges = []
      
      # Train a new tokenizer on Maltese data
      print(f"Training new tokenizer on Maltese data from {metadata_path}")
      traindf = pd.read_csv(metadata_path, sep="|")
      texts = traindf['text'].to_list()
      
      # Initialize BPE with existing vocab and merges
      bpe_model = BPE(vocab=existing_vocab, merges=existing_merges)
      new_tokenizer = Tokenizer(bpe_model)
      new_tokenizer.pre_tokenizer = Whitespace()
      trainer = BpeTrainer(
          vocab_size=extended_vocab_size,
          special_tokens=[f"[{language}]"],
          initial_alphabet=set(),  # Start with empty alphabet to respect existing vocab
          show_progress=True
      )
      
      # Get tokens that can't be encoded by existing tokenizer
      print("Analyzing new tokens...")
      new_tokens = set()
      for text in tqdm(texts, desc="Processing texts"):
          try:
              encoded = existing_tokenizer.encode(text)
              # Check if any unknown tokens were replaced with UNK
              if encoded.tokens.count("[UNK]") > 0:
                  # Only collect the actual unknown parts
                  for word in text.strip().split():
                      if existing_tokenizer.encode(word).tokens.count("[UNK]") > 0:
                          new_tokens.add(word)
          except Exception:
              # If encoding fails, the text contains new tokens
              for word in text.strip().split():
                  try:
                      existing_tokenizer.encode(word)
                  except:
                      new_tokens.add(word)
      
      if new_tokens:
          print(f"Found {len(new_tokens)} genuinely new tokens/subwords")
          
          # Train BPE only on new tokens to get appropriate merges
          new_tokenizer.train_from_iterator(
              new_tokens,
              trainer=trainer
          )
          
          # Get only the new vocab entries
          new_vocab = new_tokenizer.get_vocab()
          truly_new_tokens = {
              token: idx for token, idx in new_vocab.items()
              if token not in existing_vocab
          }
          
          if truly_new_tokens:
              print(f"Adding {len(truly_new_tokens)} new tokens to vocabulary")
              existing_tokenizer.add_tokens(list(truly_new_tokens.keys()))
      
      # Add language token if not present
      if f"[{language}]" not in existing_vocab:
          existing_tokenizer.add_special_tokens([f"[{language}]"])
          print(f"Added special token: [{language}]")
      
      # Verify no existing token IDs were modified
      final_vocab = existing_tokenizer.get_vocab()
      for token, old_id in existing_vocab.items():
          if final_vocab[token] != old_id:
              raise RuntimeError(f"Token ID changed: {token} from {old_id} to {final_vocab[token]}")
      
      # Save extended tokenizer
      existing_tokenizer.save(tokenizer_json_path)
      print(f"Extended tokenizer saved to {tokenizer_json_path}")
      
      # Update config and resize model embeddings
      adjust_config(output_path, language, existing_tokenizer.get_vocab_size())
      resize_xtts_checkpoint_embeddings(output_path, existing_tokenizer.get_vocab_size())
      print("Vocabulary extension complete.")
      
      del existing_tokenizer, new_tokenizer
      torch.cuda.empty_cache()
      gc.collect()



def extend_tokenizer_claude(output_path: str, metadata_path: str, language: str, extended_vocab_size: int = 10_000):
  """
  Safely extends the XTTS tokenizer by adding new tokens from the provided metadata file.
  Preserves existing token IDs and BPE merges while adding only genuinely new tokens.
  """
  import pandas as pd
  from tokenizers import Tokenizer
  from tokenizers.pre_tokenizers import Whitespace
  from tokenizers.models import BPE
  from tokenizers.trainers import BpeTrainer
  from tqdm import tqdm
  import tempfile

  root = os.path.join(output_path, "")
  tokenizer_json_path = os.path.join(root, "vocab.json")
  if not os.path.exists(tokenizer_json_path):
    raise FileNotFoundError(f"vocab.json not found at {tokenizer_json_path}")

  print(f"Loading existing tokenizer from {tokenizer_json_path}")
  existing_tokenizer = Tokenizer.from_file(tokenizer_json_path)
  existing_vocab = existing_tokenizer.get_vocab()
  
  # Work with the existing tokenizer directly
  print(f"Reading texts from {metadata_path}")
  traindf = pd.read_csv(metadata_path, sep="|")
  texts = traindf['text'].to_list()
  
  # First add the language token if needed
  if f"[{language}]" not in existing_vocab:
      existing_tokenizer.add_special_tokens([f"[{language}]"])
      print(f"Added special token: [{language}]")
  
  # Find tokens that can't be encoded with current vocabulary
  print("Analyzing new tokens...")
  new_tokens = set()
  for text in tqdm(texts, desc="Processing texts"):
      # Split into words first to handle each word separately
      for word in text.strip().split():
          try:
              encoded = existing_tokenizer.encode(word)
              # Only add words that produce UNK tokens
              if "[UNK]" in encoded.tokens:
                  new_tokens.add(word)
          except Exception:
              new_tokens.add(word)
  
  if new_tokens:
      print(f"Found {len(new_tokens)} new tokens that cannot be encoded")
      # Add new tokens one by one
      added_count = existing_tokenizer.add_tokens(list(new_tokens))
      print(f"Added {added_count} new tokens to vocabulary")
      
      # Get tokens that can't be encoded by existing tokenizer
      print("Analyzing new tokens...")
      new_tokens = set()
      for text in tqdm(texts, desc="Processing texts"):
          try:
              encoded = existing_tokenizer.encode(text)
              # Check if any unknown tokens were replaced with UNK
              if encoded.tokens.count("[UNK]") > 0:
                  # Only collect the actual unknown parts
                  for word in text.strip().split():
                      if existing_tokenizer.encode(word).tokens.count("[UNK]") > 0:
                          new_tokens.add(word)
          except Exception:
              # If encoding fails, the text contains new tokens
              for word in text.strip().split():
                  try:
                      existing_tokenizer.encode(word)
                  except:
                      new_tokens.add(word)
      
      # Verify no existing token IDs were modified
      final_vocab = existing_tokenizer.get_vocab()
      for token, old_id in existing_vocab.items():
          if final_vocab[token] != old_id:
              raise RuntimeError(f"Token ID changed: {token} from {old_id} to {final_vocab[token]}")
      
      # Save extended tokenizer
      existing_tokenizer.save(tokenizer_json_path)
      print(f"Extended tokenizer saved to {tokenizer_json_path}")
      
      # Update config and resize model embeddings
      adjust_config(output_path, language, existing_tokenizer.get_vocab_size())
      resize_xtts_checkpoint_embeddings(output_path, existing_tokenizer.get_vocab_size())
      print("Vocabulary extension complete.")
      
      del existing_tokenizer
      torch.cuda.empty_cache()
      gc.collect()



# ========================================================================================================
# =============================================== OLD CODE ===============================================
# ========================================================================================================
def extend_tokenizer_base(output_path: str, metadata_path: str, language: str, extended_vocab_size: int = 10_000):
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




# ==================================================================================
# ============================== ChatGPT Solution ==================================
# ==================================================================================
import os
import json
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from collections import Counter

def safe_extend_tokenizer(output_path, metadata_csv, language, min_frequency=2):
    """
    tokenizer_path: path to existing tokenizer JSON (tokenizer.json or vocab.json)
    metadata_csv: CSV with column 'text' (pipe-separated metadata used in the repo)
    language: language code, e.g. 'mt'
    min_frequency: only add tokens seen >= this threshold
    """

    
    tokenizer_json_path = os.path.join(output_path, "vocab.json")
    if not os.path.exists(tokenizer_json_path):
      raise FileNotFoundError(f"vocab.json not found at {tokenizer_json_path}")

    # load tokenizer
    tok = Tokenizer.from_file(tokenizer_json_path)
    tok.pre_tokenizer = Whitespace()

    # read texts
    df = pd.read_csv(metadata_csv, sep="|", usecols=["text"])
    texts = df["text"].astype(str).tolist()

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

    # add language special token if missing
    lang_tok = f"[{language}]"
    if tok.token_to_id(lang_tok) is None:
        tok.add_special_tokens([lang_tok])
        print(f"Added special token {lang_tok}")

    # Save tokenizer (save tokenizer.json and a copy as vocab.json for XTTS compatibility)
    out_dir = os.path.dirname(output_path)
    out_name = os.path.join(out_dir, "tokenizer.json")
    tok.save(out_name)
    # many pipelines expect a 'vocab.json' or tokenizers accept tokenizer.json — keep a copy
    shutil.copy2(out_name, os.path.join(out_dir, "vocab.json"))
    print(f"Tokenizer saved to {out_name} and copied to vocab.json")
    return tok




# ==========================================================================================================
# =============================================== ALPHA CODE ===============================================
# ==========================================================================================================


def combine_tokenizers(old_tokenizer, new_tokenizer, save_dir):
    # Load both the json files, take the union, and store it
    json1 = json.load(open(os.path.join(old_tokenizer, 'vocab.json')))
    json2 = json.load(open(os.path.join(new_tokenizer, 'vocab.json')))

    # Create a new vocabulary
    new_vocab = {}
    idx = 0
    for word in json1.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    # Add words from second tokenizer
    for word in json2.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    # Make the directory if necessary
    os.makedirs(save_dir, exist_ok=True)

    # Save the vocab
    with open(os.path.join(save_dir, 'vocab.json'), 'w') as fp:
        json.dump(new_vocab, fp, ensure_ascii=False)

    # Merge the two merges file. Don't handle duplicates here
    # Concatenate them, but ignore the first line of the second file
    os.system('cat {} > {}'.format(os.path.join(old_tokenizer, 'merges.txt'), os.path.join(save_dir, 'merges.txt')))
    os.system('tail -n +2 -q {} >> {}'.format(os.path.join(new_tokenizer, 'merges.txt'), os.path.join(save_dir, 'merges.txt')))


def extend_tokenizer_alpha(output_path: str, metadata_path: str, language: str, extended_vocab_size: int = 10_000):
    root = os.path.join(output_path, "")
    # save seperately vocab, merges
    existing_tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    old_tokenizer_path = os.path.join(root, "old_tokenizer/")
    os.makedirs(old_tokenizer_path, exist_ok=True)
    existing_tokenizer.model.save(old_tokenizer_path)

    # train new tokenizer
    traindf = pd.read_csv(metadata_path, sep="|")
    texts = traindf.text.to_list()

    new_tokenizer = Tokenizer(BPE())
    new_tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=[f"[{language}]"], vocab_size=extended_vocab_size)
    new_tokenizer.train_from_iterator(iter(texts), trainer=trainer)
    new_tokenizer.add_special_tokens([f"[{language}]"])

    new_tokenizer_path = os.path.join(root, "new_tokenizer/")
    os.makedirs(new_tokenizer_path, exist_ok=True)
    new_tokenizer.model.save(new_tokenizer_path)

    merged_tokenizer_path = os.path.join(root, "merged_tokenizer/")
    combine_tokenizers(
        old_tokenizer_path,
        new_tokenizer_path,
        merged_tokenizer_path
    )

    tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    tokenizer.model = tokenizer.model.from_file(os.path.join(merged_tokenizer_path, 'vocab.json'), os.path.join(merged_tokenizer_path, 'merges.txt'))
    tokenizer.add_special_tokens([f"[{language}]"])

    tokenizer.save(os.path.join(root, "vocab.json"))

    os.system(f'rm -rf {old_tokenizer_path} {new_tokenizer_path} {merged_tokenizer_path}')







if __name__ == "__main__":
  from parsers import create_tokenizer_extension_parser
  parser = create_tokenizer_extension_parser()
  args = parser.parse_args()

  extend_tokenizer_alpha(
    output_path=args.output_path,
    metadata_path=args.metadata_path,
    language=args.language,
    extended_vocab_size=args.extended_vocab_size
  )

  # safe_extend_tokenizer(
  #    output_path=args.output_path,
  #    metadata_csv=args.metadata_path,
  #    language=args.language
  # )

  # extend_tokenizer_claude(
  #   output_path=args.output_path,
  #   metadata_path=args.metadata_path,
  #   language=args.language,
  #   extended_vocab_size=args.extended_vocab_size
  # )


# from tokenizers import Tokenizer
# import torch, json

# tok = Tokenizer.from_file("/path/to/vocab.json")
# print("tokenizer size:", tok.get_vocab_size())

# ckpt = torch.load("/path/to/best_model.pth", map_location="cpu")
# state = ckpt.get("model", ckpt.get("state_dict", ckpt))
# # find embedding key (one of the keys below)
# for k in ["gpt.text_embedding.weight","gpt.gpt.wte.weight","gpt.gpt_inference.embeddings.weight"]:
#     if k in state:
#         print(k, state[k].shape)



# for t in ["[mt]","[en]","[STOP]","hello"]:
#   print(t, tok.token_to_id(t), tok.encode(t).tokens)