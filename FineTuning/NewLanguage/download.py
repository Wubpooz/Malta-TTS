import compatibility

import os
from TTS.utils.manage import ModelManager

def download(output_path: str, version: str = "main", custom_model: str = ""):
  """Download the XTTS model files and prepare the environment for training or inference.

  Args:
      output_path (str): Path to the output directory. They will be saved in a subdirectory named "models/<version>" within this path.
      version (str): Version of the XTTS model to download. Default is "main".
      custom_model (str): Path to a custom model checkpoint (.pth file) to use instead of the default XTTS model.
  Raises:
      ValueError: If the custom model is not a valid .pth file.
  """
  DVAE_CHECKPOINT_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/dvae.pth"
  MEL_NORM_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/mel_stats.pth"
  TOKENIZER_FILE_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/vocab.json"
  XTTS_CHECKPOINT_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/model.pth"
  XTTS_CONFIG_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/config.json"
  XTTS_SPEAKERS_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/speakers_xtts.pth"

  # TODO move files there ?
  # CHECKPOINTS_OUT_PATH = os.path.join(output_path, "models", f"{version}")
  # os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)
  
  DVAE_CHECKPOINT = os.path.join(output_path, os.path.basename(DVAE_CHECKPOINT_LINK))
  MEL_NORM_FILE = os.path.join(output_path, os.path.basename(MEL_NORM_LINK))
  XTTS_CHECKPOINT = os.path.join(output_path, os.path.basename(XTTS_CHECKPOINT_LINK))
  TOKENIZER_FILE = os.path.join(output_path, os.path.basename(TOKENIZER_FILE_LINK))
  XTTS_CONFIG_FILE = os.path.join(output_path, os.path.basename(XTTS_CONFIG_LINK))
  XTTS_SPEAKERS_FILE = os.path.join(output_path, os.path.basename(XTTS_SPEAKERS_LINK))

  if custom_model:
    if os.path.isfile(custom_model) and custom_model.endswith('.pth'):
      XTTS_CHECKPOINT = custom_model
      print(f" > Loading custom model: {XTTS_CHECKPOINT}")
    else:
      raise ValueError(f"Error: The specified custom model is not a valid .pth file: {custom_model}")

  if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], output_path, progress_bar=True)

  print(f" > Downloading XTTS v-{version} files...")
  if not os.path.isfile(XTTS_CHECKPOINT):  # don't download again if the checkpoint exists or when using a custom model
    print(" > Downloading XTTS checkpoint...")
    ModelManager._download_model_files([XTTS_CHECKPOINT_LINK], output_path, progress_bar=True) # private API
  if not os.path.isfile(TOKENIZER_FILE):
    print(" > Downloading XTTS tokenizer...")
    ModelManager._download_model_files([TOKENIZER_FILE_LINK], output_path, progress_bar=True)
  if not os.path.isfile(XTTS_CONFIG_FILE):
    print(" > Downloading XTTS config file...")
    ModelManager._download_model_files([XTTS_CONFIG_LINK], output_path, progress_bar=True)
  if not os.path.isfile(XTTS_SPEAKERS_FILE):
    print(" > Downloading XTTS speakers file...")
    ModelManager._download_model_files([XTTS_SPEAKERS_LINK], output_path, progress_bar=True)
    # or Create an empty speakers file
    # torch.save({}, XTTS_SPEAKERS_FILE)

  print(" > XTTS model files downloaded successfully!")


  return MEL_NORM_FILE, DVAE_CHECKPOINT, XTTS_CHECKPOINT, TOKENIZER_FILE


if __name__ == "__main__":
  from parsers import create_download_parser

  parser = create_download_parser()
  args = parser.parse_args()
  download(
    output_path=args.output_path,
    version=args.version,
    custom_model=args.custom_model
  )