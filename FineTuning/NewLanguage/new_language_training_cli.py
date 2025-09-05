import os

from download import download
from tokenizer_extension import extend_tokenizer_with_validation
from trainingGPT import train_gpt
from parsers import create_xtts_trainer_parser

def validate_args(args) -> None:
  """
  Validate the command line arguments.
  Raises:
    ValueError: If any of the arguments are invalid.
  """
  if not os.path.exists(args.output_path):
    print(f"ERROR: Output path does not exist: {args.output_path}")
    raise ValueError("Invalid output path.")
  if not args.language:
    print("ERROR: Language must be specified.")
    raise ValueError("Invalid language.")
  if not args.is_download:
    if not args.mel_norm_file:
      print("ERROR: Mel normalization file must be specified.")
      raise ValueError("Invalid mel normalization file.")
    if not args.dvae_checkpoint:
      print("ERROR: DVAED checkpoint file must be specified.")
      raise ValueError("Invalid DVAED checkpoint file.")
    if not args.xtts_checkpoint:
      print("ERROR: XTTS checkpoint file must be specified.")
      raise ValueError("Invalid XTTS checkpoint file.")
    if not args.tokenizer_file:
      print("ERROR: Tokenizer file must be specified.")
      raise ValueError("Invalid tokenizer file.")
    if not args.metadatas or len(args.metadatas) == 0:
      print("ERROR: Metadata files must be specified.")
      raise ValueError("Invalid metadata files.")
    else:
      # Normalize metadata format
      normalized = []
      for m in args.metadatas:
        if isinstance(m, str):
          parts = m.split(",")
          if len(parts) != 3:
            raise ValueError(f"Metadata must be train,eval,lang but got: {m}")
          normalized.append(tuple(parts))
        elif isinstance(m, (tuple, list)) and len(m) == 3:
          normalized.append(tuple(m))
        else:
          raise ValueError(f"Invalid metadata format: {m}")
      args.metadatas = normalized


def training(args) -> tuple:
  """
  Train a new language model.
  Arguments (in args):
      output_path (str): The output path for the training.
      is_download (bool): Whether to download the model files. Default is True.
      version (str): The version of the model to download. Default is "main".
      language (str): The language to train the model on.
      mel_norm_file (str): The mel normalization file.
      dvae_checkpoint (str): The DVAED checkpoint file.
      xtts_checkpoint (str): The XTTS checkpoint file.
      tokenizer_file (str): The tokenizer file.
      metadatas (list): The metadata files in the format [train_csv_path, eval_csv_path, language].
      num_epochs (int): The number of training epochs. Default is 10.
      batch_size (int): The batch size for training. Default is 3.
      grad_acumm (int): The gradient accumulation steps. Default is 84.
      max_audio_length (int): The maximum audio length. Default is 255995s.
      max_text_length (int): The maximum text length. Default is 200.
      weight_decay (float): The weight decay for the optimizer. Default is 1e-2.
      lr (float): The learning rate for the optimizer. Default is 5e-6.
      save_step (int): The number of steps between model saves. Default is 10000.
      custom_model (str): The path to a custom model checkpoint (.pth file).
      multi_gpu (bool): Whether to use multi-GPU training. Default is False.
      optimizations (bool): Whether to enable optimizations for training. Default is False.
      tf32 (bool): Whether to enable TF32 for training. Default is False.
      forgetting_mitigation (str): The method to mitigate forgetting during training. Between "none", "LORA", and "FREEZE". Default is "LORA".
  Returns:
    tuple: A tuple containing the updated XTTS checkpoint, vocabulary size, configuration, training output path, and speaker reference.
  Raises:
    ValueError: If any of the arguments are invalid.
  """
  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path, exist_ok=True)

  step = 1
  if args.is_download:
    print(f"Step {step}: Downloading XTTS base model files.")
    mel_norm_file, dvae_checkpoint, xtts_checkpoint, tokenizer_file, config, speakers_file = download(
      output_path=args.output_path,
      version=args.version
    )
    step += 1
  else:
    mel_norm_file = args.mel_norm_file
    dvae_checkpoint = args.dvae_checkpoint
    xtts_checkpoint = args.xtts_checkpoint
    tokenizer_file = args.tokenizer_file
    config = args.config

  training_metadata_path = args.metadatas[0][0]
  print(f"Using training metadata for tokenizer extension: {training_metadata_path}")
  #TODO check if already extended and if so don't do anything (if model & model_backup, if tokenizer.vocab>68.. and if config.mt exists)
  print(f"Step {step}: Extending the XTTS tokenizer with the new language.")
  vocab_size, xtts_checkpoint, tokenizer_file = extend_tokenizer_with_validation(
    output_path=args.output_path,
    xtts_checkpoint=xtts_checkpoint,
    tokenizer_file=tokenizer_file,
    config_path=config,
    metadata_path=training_metadata_path,
    language=args.language,
    min_frequency=args.min_frequency,
    max_new_tokens=args.max_new_tokens,
  )
  print(f"Extended vocabulary size: {vocab_size}")
  step += 1

  print(f"Step {step}: Starting GPT training.")
  xtts_checkpoint, xtts_vocab, config, trainer_out_path, speaker_ref = train_gpt(
    metadatas=args.metadatas,
    language=args.language,
    mel_norm_file=mel_norm_file if args.is_download else args.mel_norm_file,
    dvae_checkpoint=dvae_checkpoint if args.is_download else args.dvae_checkpoint,
    xtts_checkpoint=xtts_checkpoint,
    tokenizer_file=tokenizer_file,
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

  print(f"Checkpoint saved in dir: {trainer_out_path}. Speaker reference saved in: {speaker_ref}")
  return xtts_checkpoint, xtts_vocab, config, trainer_out_path, speaker_ref


def main(args):
  validate_args(args)

  return training(args)


if __name__ == "__main__":
  parser = create_xtts_trainer_parser()
  args = parser.parse_args()

  main(args)  