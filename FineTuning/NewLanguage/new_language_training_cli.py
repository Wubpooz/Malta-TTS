import os
import json

from download import download
from tokenizer_extension import extend_tokenizer_with_validation
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

  #TODO check if already extended and if so don't do anything (if model & model_backup, if tokenizer.vocab>68.. and if config.mt exists)
  print(f"Step {step}: Extending the XTTS tokenizer with the new language.")
  vocab_size = extend_tokenizer_with_validation(
    output_path=args.output_path,
    metadata_path=args.metadata_path,
    language=args.language,
    vocab_size=5000,
    min_frequency=2,
    max_new_tokens=8000
  )
  print(f"Extended vocabulary size: {vocab_size}")
  step += 1


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
    tf32=args.tf32,
    forgetting_mitigation=args.forgetting_mitigation
  )
  
  print(f"Checkpoint saved in dir: {trainer_out_path}")