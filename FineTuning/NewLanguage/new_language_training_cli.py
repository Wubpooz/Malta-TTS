from trainingGPT import train_gpt
from tokenizer_extension import extend_tokenizer
from tokenizer_extension import adjust_config
from download import download
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

  if args.is_tokenizer_extension:
    print(f"Step {step}: Extending the XTTS tokenizer with the new language.")
    extend_tokenizer(
      output_path=args.output_path,
      metadata_path=args.metadata_path,
      language=args.language,
      extended_vocab_size=args.extended_vocab_size
    )
  else:
    print(f"Step {step}: Adjusting the config file.")
    adjust_config(
      root=args.output_path,
      language=args.language,
      vocab_size=args.extended_vocab_size
    )
  step += 1  

  print(f"Step {step}: Starting GPT training.")
  xtts_checkpoint, xtts_vocab, config, trainer_out_path, speaker_ref = train_gpt(
    metadatas=args.metadatas,
    language=args.language,
    mel_norm_file=mel_norm_file if args.is_download else args.mel_norm_file,
    dvae_checkpoint=dvae_checkpoint if args.is_download else args.dvae_checkpoint,
    xtts_checkpoint=xtts_checkpoint if args.is_download else args.xtts_checkpoint,
    tokenizer_file=tokenizer_file if args.is_download else args.tokenizer_file,
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




# Phonetic transcription
# A rule-based script to transcribe Maltese text into IPA notation. An example is shown below.

# >> from masri.transcribe.g2p import text2phon
# >> print(text2phon("Ilbieraħ mort s'Għawdex"))
# ɪlbɪːrɐh mɔrt sɐʊdɛʃ
# Numbers to words
# An extension of num2words for the Maltese language. An example is shown below.

# >> from masri.transcribe.num2text import num2text
# >> print(num2text(301000))
# tliet mitt elf u  elf
