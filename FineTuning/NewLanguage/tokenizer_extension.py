import os
import json
import torch
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

def _merge_tokenizers_preserve_ids(old_tokenizer_path, new_tokenizer_path, output_path):
  """
  Merges two vocabularies, preserving the token IDs from the old tokenizer
  and adding new tokens from the new tokenizer.
  """
  print(f"Merging tokenizers from {old_tokenizer_path} and {new_tokenizer_path} into {output_path}")
  with open(os.path.join(old_tokenizer_path, 'vocab.json')) as f:
    old_vocab = json.load(f)
    print(f"Old tokenizer vocabulary size: {len(old_vocab)}")
  with open(os.path.join(new_tokenizer_path, 'vocab.json')) as f:
    new_vocab = json.load(f)
    print(f"New tokenizer vocabulary size: {len(new_vocab)}")

  combined_vocab = old_vocab.copy()
  new_id = max(old_vocab.values())
  
  # Iterate through the new vocabulary and add words not in the old one
  for word, _ in new_vocab.items():
    if word not in combined_vocab:
      new_id += 1
      combined_vocab[word] = new_id

  print(f"Combined vocabulary size: {len(combined_vocab)}")
  os.makedirs(output_path, exist_ok=True)
  with open(os.path.join(output_path, 'vocab.json'), 'w') as fp:
    json.dump(combined_vocab, fp, ensure_ascii=False, indent=2)
  print(f"Combined vocabulary saved to {os.path.join(output_path, 'vocab.json')}")

  return combined_vocab


def extend_tokenizer(output_path: str, metadata_path: str, language: str, extended_vocab_size: int = 100_000, version: str = "main"):
  """Extends the XTTS tokenizer with new vocabulary from the provided metadata file.
  This function combines the existing tokenizer with a new tokenizer trained on the provided metadata.
  It saves the new tokenizer in a specified directory and updates the vocabulary to include new tokens.
  Args:
      output_path (str): Path to the output directory where the tokenizer files will be saved.
      metadata_path (str): Path to the metadata file containing training data.
      language (str): Language code for the new language to be added.
      extended_vocab_size (int): Desired size of the extended vocabulary. Default is 100_000.
  """
  root = os.path.join(output_path, "")
  
  old_tokenizer_path = os.path.join(root, "old_tokenizer/")
  new_tokenizer_path = os.path.join(root, "new_tokenizer/")
  merged_tokenizer_path = os.path.join(root, "merged_tokenizer/")

  os.makedirs(old_tokenizer_path, exist_ok=True)
  existing_tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
  existing_tokenizer.model.save(old_tokenizer_path)
  print(f"Original tokenizer loaded with {len(existing_tokenizer.get_vocab())} tokens.")

  traindf = pd.read_csv(metadata_path, sep="|")
  texts = traindf['text'].to_list()
  new_tokenizer = Tokenizer(BPE())
  new_tokenizer.pre_tokenizer = Whitespace() # type: ignore
  trainer = BpeTrainer(special_tokens=[f"[{language}]"], vocab_size=extended_vocab_size) # type: ignore
  print(f"Training new tokenizer with {len(texts)} texts...")
  new_tokenizer.train_from_iterator(iter(texts), trainer=trainer)
  new_tokenizer.add_special_tokens([f"[{language}]"])

  print(f"New tokenizer trained with {len(new_tokenizer.get_vocab())} tokens.")
  os.makedirs(new_tokenizer_path, exist_ok=True)
  new_tokenizer.model.save(new_tokenizer_path)

  _merge_tokenizers_preserve_ids(old_tokenizer_path, new_tokenizer_path, merged_tokenizer_path)

  # 4. Now, create the final tokenizer by combining the merged vocab with the new merges.txt
  merged_vocab_file = os.path.join(merged_tokenizer_path, 'vocab.json')
  new_merges_file = os.path.join(new_tokenizer_path, 'merges.txt')

  final_tokenizer = Tokenizer(BPE(vocab=merged_vocab_file, merges=new_merges_file))
  final_tokenizer.pre_tokenizer = Whitespace() # type: ignore
  final_tokenizer.add_special_tokens([f"[{language}]"])

  # 5. Overwrite the original vocab.json with the new, extended one
  final_tokenizer.save(os.path.join(root, "vocab.json"))

  # Clean up temporary files
  os.system(f'rm -rf {old_tokenizer_path} {new_tokenizer_path} {merged_tokenizer_path}')

  print(f"Tokenizer has been successfully extended and saved to {os.path.join(root, 'vocab.json')}")

  print("Updating the XTTS checkpoint...")
  clean_xtts_checkpoint(output_path)

  print("Updating the XTTS config file...")
  adjust_config(output_path, version, language)



def adjust_config(output_path: str, version: str, language: str):
  """Adjust the XTTS configuration file to include the new language.
  Args:
      output_path (str): Path to the output directory where the config file is located (it will be appended with "/models/{version}/config.json"). 
      version (str): Version of the XTTS model.
      language (str): Language code for the new language to be added.
  """
  config_path = os.path.join(output_path, "config.json")
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at {config_path}. Please ensure the path is correct.")
  with open(config_path, "r") as f:
    config = json.load(f)
  config["languages"] += [language]
  with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)
  print(f"Updated config file saved to {config_path}. Added new language: {language}")



def clean_xtts_checkpoint(original_path: str):
  """Removes incompatible layers after tokenizer extension.
  Args:
      original_path (str): Path to the original XTTS checkpoint directory.
  """
  xtts_checkpoint_path = os.path.join(original_path, "model.pth")
  if not os.path.exists(original_path) or not os.path.exists(xtts_checkpoint_path):
    raise FileNotFoundError(f"Original checkpoint file not found at {original_path}. Please ensure the path is correct.")

  print(f"Cleaning checkpoint: {xtts_checkpoint_path}")

  checkpoint = torch.load(xtts_checkpoint_path, map_location="cpu")

  for key in ["gpt.text_embedding.weight", "gpt.text_head.weight", "gpt.text_head.bias"]:
    if key in checkpoint["model"]:
      print(f"Removing incompatible layer: {key}")
      del checkpoint["model"][key]

  if os.path.exists(xtts_checkpoint_path):
    os.rename(xtts_checkpoint_path, xtts_checkpoint_path + ".old")

  torch.save(checkpoint, xtts_checkpoint_path)
  print(f"Cleaned checkpoint saved.")
  return xtts_checkpoint_path



if __name__ == "__main__":
  from parsers import create_tokenizer_extension_parser
  parser = create_tokenizer_extension_parser()
  args = parser.parse_args()

  extend_tokenizer(
    output_path=args.output_path,
    metadata_path=args.metadata_path,
    language=args.language,
    version=args.version,
    extended_vocab_size=args.extended_vocab_size
  )