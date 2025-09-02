import os
import io
import shutil
import librosa
import datasets
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from huggingface_hub import HfApi, HfFolder, Repository, create_repo, login
from datasets import load_dataset, Audio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split

# python FineTuning\NewLanguage\prepare_maltese_dataset.py --input_dir "C:\\Users\\mathi\\Downloads\\MASRI_HEADSET_v2" --output_dir "C:\\Users\\mathi\\Downloads\\MASRI_HEADSET_HF" --test_size 0.2 --upload_to_hf --dataset_name "Bluefir/maltese-headset-v2_test"

# ========================================================================================================
# ===================================== Dataset Preparation ==============================================
# ========================================================================================================
def prepare_dataset(input_dir: str, output_dir: str, test_size: float = 0.1) -> str:
  """
  Prepares the MASRI-HEADSET CORPUS v2 dataset by splitting it into train/test
  and creating a directory structure suitable for Hugging Face.
  Arguments:
    input_dir (str): Path to the input directory containing the dataset files.
    output_dir (str): Path to the output directory where the prepared dataset will be saved.
    test_size (float): Proportion of the dataset to include in the test split. Default is 0.1 (10%).
  Returns:
    str: The path to the prepared dataset directory.
  Raises:
    FileNotFoundError: If the transcription file is not found.
  """
  print("Starting dataset preparation...")
  transcriptions_file = os.path.join(input_dir, "files", "MASRI_HEADSET_v2.trans")
  if not os.path.exists(transcriptions_file):
    raise FileNotFoundError(f"Transcription file not found at: {transcriptions_file}")
  
  transcriptions = {}
  with open(transcriptions_file, "r", encoding="utf-8") as f:
    for line in tqdm(f.readlines(), desc="Parsing transcriptions"):
      parts = line.strip().split(" ", 1)
      if len(parts) == 2:
        transcriptions[parts[0]] = parts[1]

  all_data = []
  speech_dir = os.path.join(input_dir, "speech")
  for gender_dir_name in ['female', 'male']:
    gender_dir = os.path.join(speech_dir, gender_dir_name)
    if not os.path.exists(gender_dir):
      continue
    for speaker_id in tqdm(os.listdir(gender_dir), desc=f"Processing {gender_dir_name} speakers"):
      speaker_path = os.path.join(gender_dir, speaker_id)
      if os.path.isdir(speaker_path):
        for audio_file in os.listdir(speaker_path):
          if audio_file.endswith(".wav"):
            file_id = os.path.splitext(audio_file)[0]
            if file_id in transcriptions:
              audio_abs_path = os.path.join(speaker_path, audio_file)
              audio_rel_path = os.path.join("wavs", audio_file)
              all_data.append({"audio_file": audio_rel_path, "text": transcriptions[file_id], "speaker_name": speaker_id, "audio_abs_path": audio_abs_path})
  
  if not all_data:
    raise ValueError("No data found to process. Please check your input directory and file structure.")
  
  df = pd.DataFrame(all_data)
  print(f"Total files collected: {len(df)}")
  
  train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

  dataset_path = os.path.join(output_dir)
  os.makedirs(dataset_path, exist_ok=True)
  wavs_path = os.path.join(dataset_path, "wavs")
  os.makedirs(wavs_path, exist_ok=True)
  
  print("Symlinking audio files to the output directory...")
  all_files_to_link = pd.concat([train_df, test_df])
  for _, row in tqdm(all_files_to_link.iterrows(), total=len(all_files_to_link), desc="Symlinking files"):
    src = row['audio_abs_path']
    dst = os.path.join(wavs_path, os.path.basename(src))
    if not os.path.exists(dst):
      try:
        os.symlink(src, dst)
      except OSError as e:
        shutil.copy(src, dst)

  train_df[['audio_file', 'text', 'speaker_name']].to_csv(os.path.join(dataset_path, "metadata_train.csv"), sep="|", index=False)
  test_df[['audio_file', 'text', 'speaker_name']].to_csv(os.path.join(dataset_path, "metadata_test.csv"), sep="|", index=False)

  print("\nDataset preparation completed successfully!")
  print(f"Dataset saved to: {dataset_path}")
  print(f" - Train files: {len(train_df)}")
  print(f" - Test files: {len(test_df)}")
  return dataset_path



def save_to_huggingFace(dataset_path: str, dataset_name: str) -> None:
  """
  Save the dataset to the Hugging Face Hub.
  Arguments:
      dataset_path (str): Path to the dataset directory.
      dataset_name (str): Name of the dataset on the Hugging Face Hub.
  Returns:
      None
  Raises:
      Exception: If the dataset fails to load or upload.
  """
  print(f"\nLoading dataset from {dataset_path} using direct file loading...")
  try:
    # Use `load_dataset` with the CSV files directly.
    dataset_dict = load_dataset(
      'csv', 
      data_files={
        'train': os.path.join(dataset_path, "metadata_train.csv"),
        'test': os.path.join(dataset_path, "metadata_test.csv"),
      },
      delimiter="|",
      column_names=["audio_file", "text", "speaker_name"],
    )
    
    dataset_dict = dataset_dict.rename_column("speaker_name", "speaker_id")
    dataset_dict = dataset_dict.rename_column("text", "normalized_text")


    def add_features(example):
      audio_path = os.path.join(dataset_path, example['audio_file'])
      example['audio'] = audio_path

      speaker_id = example["speaker_id"]
      example['gender'] = "unknown"
      if speaker_id.startswith("F_"):
        example['gender'] = "female"
      elif speaker_id.startswith("M_"):
        example['gender'] = "male"

      try:
        info = sf.info(audio_path)
        example['duration'] = info.frames / info.samplerate
      except Exception:
        example['duration'] = 0.0                
      return example

    dataset_dict = dataset_dict.map(add_features, num_proc=1, desc="Adding extra features") # type: ignore

    # Cast the columns to the correct features
    features = datasets.Features({
      "audio": datasets.Audio(sampling_rate=22050),
      "speaker_id": datasets.Value("string"),
      "gender": datasets.Value("string"),
      "duration": datasets.Value("float32"),
      "normalized_text": datasets.Value("string"),
    })
    # Clean up the original `audio_file` column to match the expected format
    dataset_dict = dataset_dict.remove_columns("audio_file")
    dataset_dict = dataset_dict.cast(features)

    print("\nDataset loaded successfully!")
    print(dataset_dict)

    input("Do you want to upload this dataset to Hugging Face? Press Enter to continue or Ctrl+C to exit.")
    dataset_dict.push_to_hub(dataset_name)
    print("Dataset pushed to Hugging Face Hub successfully!")

  except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure 'metadata_train.csv', 'metadata_test.csv', and the 'wavs' folder are correctly placed inside the output directory.")
  except Exception as e:
    print(f"An unexpected error occurred during loading or uploading: {e}")



# =====================================================
# =============== Loading Dataset =====================
# =====================================================
def _save_and_resample_worker(example: dict, wavs_dir: str, sampling_rate: int, save_audio: bool = True):
  """Top-level worker for ProcessPoolExecutor (picklable)."""
  try:
    audio_info = example['audio']
    audio_bytes = audio_info.get('bytes') or audio_info.get('array')  # support decoded/undecoded
    audio_filename = audio_info.get('path') or example.get('audio_file')
    if not audio_bytes or not audio_filename:
      print(f"Warning: missing audio data for example {example.get('id')}")
      return None
    base_name = os.path.splitext(os.path.basename(audio_filename))[0]

    if save_audio:
      out_path = os.path.join(wavs_dir, f"{base_name}.wav")
      # read audio (bytes -> numpy array)
      if isinstance(audio_bytes, (bytes, bytearray)):
        with io.BytesIO(audio_bytes) as f:
          y, sr = sf.read(f, dtype='float32')
      else:
        y = audio_bytes
        sr = audio_info.get('sampling_rate', 22050)

      if sr != sampling_rate:
        # print(f"Warning: resampling {audio_filename} from {sr} to {sampling_rate}")
        try:
          import samplerate
          ratio = float(sampling_rate) / float(sr)
          y = samplerate.resample(y, ratio, 'sinc_best')
          sr = sampling_rate
        except Exception:
          y = librosa.resample(y.astype('float32'), orig_sr=sr, target_sr=sampling_rate)
          sr = sampling_rate

        try:
          sf.write(out_path, y, samplerate=sampling_rate)
        except Exception as e:
          print(f"Warning: failed to write resampled file {out_path}: {e}")
          # fallback: if original bytes are available and user requested to save them, write raw bytes
          if save_audio and isinstance(audio_info.get('bytes'), (bytes, bytearray)):
            try:
              raw_dst = os.path.join(wavs_dir, os.path.basename(audio_info.get('path', out_path)))
              with open(raw_dst, 'wb') as rf:
                rf.write(audio_info.get('bytes'))
              out_path = raw_dst
            except Exception as e2:
              print(f"Error: failed to save raw audio bytes for {base_name}: {e2}")
              return {'__error__': f"write_failed: {e2}", 'audio_file': None}
          else:
            return {'__error__': f"write_failed: {e}", 'audio_file': None}


    # Use LJSpeech format (extended)
    # /!\ audio_file shouldn't have extension, else fails | also they should just be filenames, the loader will add wav/ before and .wav after
    return {
      'audio_file': base_name,
      'text': example.get('normalized_text') or example.get('text') or "",
      'normalized_text': example.get('normalized_text') or example.get('text') or "",
      'speaker_name': example.get('speaker_id') or example.get('speaker_name') or ""
    }
  except Exception as e:
    return {'__error__': str(e), 'audio_file': None}


def process_split(split_name: str, csv_filename: str, output_wavs_dir: str, ds, sampling_rate: int = 22050, save_audio=False, num_workers: int = 8):
  """Process a dataset split using ProcessPoolExecutor and as_completed for streaming progress."""
  print(f"Processing split: {split_name} (workers={num_workers})")
  results = []
  futures = []
  total = len(ds[split_name])

  # Reads first file and tell if there will likely be resampling (since we can assume that all the audio files follow the same format)
  first_audio_info = ds[split_name][0].get('audio')
  if first_audio_info is not None:
    first_sampling_rate = first_audio_info.get('sampling_rate', 22050)
    if first_sampling_rate != sampling_rate:
      print(f"{split_name} split audio files are likely sampled at {first_sampling_rate}, they may need resampling to {sampling_rate}.")

  with ProcessPoolExecutor(max_workers=num_workers) as exc:
    for ex in ds[split_name]:
      futures.append(exc.submit(_save_and_resample_worker, ex, output_wavs_dir, sampling_rate, save_audio))

    with tqdm(total=total, desc=f"Resampling {split_name}") as pbar:
      for fut in as_completed(futures):
        res = fut.result()
        pbar.update(1)
        if not res or res.get('__error__'):
          if res:
            print(f"Warning: worker error: {res.get('__error__')}")
          continue
        results.append(res)

  # Save metadata
  df = pd.DataFrame(results)
  df.to_csv(os.path.join(os.path.dirname(output_wavs_dir), csv_filename), sep="|", index=False)
  print(f"Saved {len(df)} entries to {csv_filename}")


def load_and_resample(output_dir: str, dataset: str = "Bluefir/MASRI_HEADSET_v2", sampling_rate: int = 22050, num_workers: int = 16) -> None:
  """
  Load a dataset and resample its audio files.
  Arguments:
    output_dir (str): Directory to save the processed audio files.
    dataset (str): Name of the dataset to load.
    sampling_rate (int): Target sampling rate for audio files.
    num_workers (int): Number of worker threads for processing.
  Raises:
    ValueError: If the dataset is not found or cannot be loaded.
  """
  os.makedirs(output_dir, exist_ok=True)
  wavs_dir = os.path.join(output_dir, "wavs")
  os.makedirs(wavs_dir, exist_ok=True)

  print("Loading dataset from Hugging Face...")
  hf_token = os.environ.get("HF_TOKEN")
  ds = load_dataset(dataset, use_auth_token=hf_token)
  ds = ds.cast_column("audio", Audio(decode=False))

  print(f"Resampling to {sampling_rate}Hz and saving...")
  process_split("train", "metadata_train.csv", wavs_dir, ds, sampling_rate=sampling_rate, save_audio=True, num_workers=num_workers)
  process_split("test", "metadata_eval.csv", wavs_dir, ds, sampling_rate=sampling_rate, save_audio=True, num_workers=num_workers)
  print("Dataset saved!")



# ==================================================
# ==================== Utilities ===================
# ==================================================
def dataset_repartition(dataset: str = "Bluefir/MASRI_HEADSET_v2", local: bool = False) -> None:
  """
    Repartition the dataset into train and test splits. Prints the text length and audio duration statistics.
    Arguments:
      dataset (str): The name of the huggingFace dataset to repartition.
    Returns:
      None
  """
  import tempfile
  # Don't decode audio — just keep metadata
  hf_token = os.environ.get("HF_TOKEN")
  ds = load_dataset(dataset, use_auth_token=hf_token)
  ds = ds.cast_column("audio", Audio(decode=False))
  text_lengths = []
  audio_durations = []

  for split in ["train", "test"]:
    print(f"Processing split: {split}")
    for example in ds[split]:
      text_lengths.append(len(example["normalized_text"])) # type: ignore

      # Save audio bytes to temp file, read duration with soundfile
      audio_bytes = example["audio"]["bytes"] # type: ignore
      with tempfile.NamedTemporaryFile(suffix=".wav") as tmpf:
        tmpf.write(audio_bytes)
        tmpf.flush()
        with sf.SoundFile(tmpf.name) as f:
          duration = len(f) / f.samplerate
          audio_durations.append(duration)

  print(f"Text length range: {min(text_lengths)} - {max(text_lengths)} characters")
  print(f"Audio duration range: {min(audio_durations):.2f} - {max(audio_durations):.2f} seconds")
  print(f"Average text length: {sum(text_lengths)/len(text_lengths):.2f} characters")
  print(f"Average audio duration: {sum(audio_durations)/len(audio_durations):.2f} seconds")




if __name__ == "__main__":
  from parsers import create_prepare_dataset_parser
  parser = create_prepare_dataset_parser()
  args = parser.parse_args()

  dataset_path = prepare_dataset(args.input_dir, args.output_dir, args.test_size)

  if args.upload_to_hf:
    save_to_huggingFace(dataset_path, args.dataset_name)