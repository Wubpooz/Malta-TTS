import os
import torch
import torchaudio

from trainingGPT import train_gpt
from inference import inference
from tokenizer_extension import extend_tokenizer
from tokenizer_extension import adjust_config
from download import download
from parsers import create_xtts_trainer_parser


if __name__ == "__main__":
  parser = create_xtts_trainer_parser()
  args = parser.parse_args()

  # Step 1: Download the base XTTS model files.
  print("Step 1: Downloading XTTS base model files.")
  download(
    output_path=args.output_path,
    version=args.version
  )
  
  # Step 2: Extend the tokenizer for the new language.
  print("Step 2: Extending the XTTS tokenizer with the new language.")
  extend_tokenizer(
    output_path=args.output_path,
    metadata_path=args.metadata_path, #datasets/metadata_train.csv
    language=args.language,
    extended_vocab_size=args.extended_vocab_size #2000
  )
  
  # Step 3: Adjust the config file to include the new language.
  print("Step 3: Adjusting the config file.")
  adjust_config(
    output_path=args.output_path,
    version=args.version,
    language=args.language
  )

  # Step 4: Start the training process with the extended tokenizer and updated config.
  print("Step 4: Starting GPT training.")
  xtts_checkpoint, xtts_vocab, config, trainer_out_path, speaker_ref = train_gpt(
    metadatas=args.metadatas,
    output_path=args.output_path,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    grad_acumm=args.grad_acumm,
    weight_decay=args.weight_decay,
    lr=args.lr,
    max_text_length=args.max_text_length,
    max_audio_length=args.max_audio_length,
    save_step=args.save_step
  )
  
  print(f"Checkpoint saved in dir: {trainer_out_path}")

  # Step 5: Run inference on the trained model.
  # run_inference = input("Do you want to run inference? (y/n): ").strip().lower()
  # inference_text = input("Enter the text for inference (or leave empty to use default): ").strip() # Hija test tal-mudell tat-taħdit il-ġdid tiegħi, il-lingwa Maltija hija interessanti! Esperimenti u testijiet huma importanti biex niskopru l-possibbiltajiet tat-taħdit.
  # if run_inference == 'y':
  #   print("Running inference...")
  #   audio = inference(
  #     xtts_checkpoint=xtts_checkpoint,
  #     xtts_config=config,
  #     xtts_vocab=xtts_vocab,
  #     tts_text=inference_text,
  #     speaker_audio_file=speaker_ref,
  #     lang=args.language
  #   )
  #   print("Inference completed!")
  #   torchaudio.save(os.path.join(trainer_out_path, "output_maltese.wav"), audio, 24000)

  #   try:
  #     from IPython.display import Audio
  #     Audio(audio, rate=24000) # Play audio (for Jupyter Notebook)
  #   except ImportError:
  #     print("IPython not available, audio playback not supported in this environment.")
  # else:
  #   print("Skipping inference. You can run it later by calling the `inference()` function.")






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
