import os
import json

from download import download
from tokenizer_extension import extend_tokenizer
from trainingGPT import train_gpt
from parsers import create_xtts_trainer_parser


if __name__ == "__main__":
  parser = create_xtts_trainer_parser()
  args = parser.parse_args()

  step = 1
  if args.is_download:
    print(f"Step {step}: Downloading XTTS base model files.")
    mel_norm_file, dvae_checkpoint, xtts_checkpoint, tokenizer_file = download(
      output_path=args.output_path,
      version=args.version
    )
    step += 1

  print(f"Step {step}: Extending the XTTS tokenizer with the new language.")
  vocab_size = extend_tokenizer(
    output_path=args.output_path,
    metadata_path=args.metadata_path,
    language=args.language
  )
  print(f"Extended vocabulary size: {vocab_size}")
  step += 1


  config_path = os.path.join(args.output_path, "config.json")
  if os.path.exists(config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
      config = json.load(f)
    config_vocab_size = config.get("model_args", {}).get("gpt_number_text_tokens", "NOT_FOUND")
    print(f"Config vocab size: {config_vocab_size}")
    
    if config_vocab_size != vocab_size:
      print(f"WARNING: Vocab size mismatch! Tokenizer: {vocab_size}, Config: {config_vocab_size}")
      config["model_args"]["gpt_number_text_tokens"] = vocab_size
      with open(config_path, 'w', encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
      print(f"Fixed config vocab size to {vocab_size}")

  print(f"Step {step}: Starting GPT training.")
  
  updated_xtts_checkpoint = os.path.join(args.output_path, "model.pth")
  updated_tokenizer_file = os.path.join(args.output_path, "vocab.json")
  
  if not os.path.exists(updated_xtts_checkpoint):
    print(f"ERROR: Updated checkpoint not found at {updated_xtts_checkpoint}")
    exit(1)
  if not os.path.exists(updated_tokenizer_file):
    print(f"ERROR: Updated tokenizer not found at {updated_tokenizer_file}")  
    exit(1)
    
  print(f"Using updated checkpoint: {updated_xtts_checkpoint}")
  print(f"Using updated tokenizer: {updated_tokenizer_file}")
  print(f"Using vocab size: {vocab_size}")

  xtts_checkpoint, xtts_vocab, config, trainer_out_path, speaker_ref = train_gpt(
    metadatas=args.metadatas,
    language=args.language,
    mel_norm_file=mel_norm_file if args.is_download else args.mel_norm_file,
    dvae_checkpoint=dvae_checkpoint if args.is_download else args.dvae_checkpoint,
    xtts_checkpoint=updated_xtts_checkpoint,
    tokenizer_file=updated_tokenizer_file,
    vocab_size=vocab_size,
    output_path=args.output_path,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    grad_acumm=args.grad_acumm,
    weight_decay=args.weight_decay,
    lr=args.lr,
    max_text_length=args.max_text_length,
    max_audio_length=args.max_audio_length,
    save_step=args.save_step,
    optimizations=args.optimizations,
    tf32=args.tf32
  )
  
  print(f"Checkpoint saved in dir: {trainer_out_path}")