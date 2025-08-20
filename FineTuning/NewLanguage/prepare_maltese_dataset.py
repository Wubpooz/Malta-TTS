import os
import datasets
import pandas as pd
import soundfile as sf
import shutil
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# python FineTuning\NewLanguage\prepare_maltese_dataset.py --input_dir "C:\\Users\\mathi\\Downloads\\MASRI_HEADSET_v2" --output_dir "C:\\Users\\mathi\\Downloads\\MASRI_HEADSET_HF" --test_size 0.2 --upload_to_hf --dataset_name "Bluefir/maltese-headset-v2_test"

# ========================================================================================================
# ============================== Dataset Preparation and CLI Logic =======================================
# ========================================================================================================
def prepare_dataset(input_dir, output_dir, test_size):
  """
  Prepares the MASRI-HEADSET CORPUS v2 dataset by splitting it into train/test
  and creating a directory structure suitable for Hugging Face.
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


if __name__ == "__main__":
  from parsers import create_prepare_dataset_parser
  parser = create_prepare_dataset_parser()
  args = parser.parse_args()

  final_dataset_path = prepare_dataset(args.input_dir, args.output_dir, args.test_size)

  if args.upload_to_hf:
    print(f"\nLoading dataset from {final_dataset_path} using direct file loading...")
    try:
      # Use `load_dataset` with the CSV files directly.
      dataset_dict = load_dataset(
        'csv', 
        data_files={
          'train': os.path.join(final_dataset_path, "metadata_train.csv"),
          'test': os.path.join(final_dataset_path, "metadata_test.csv"),
        },
        delimiter="|",
        column_names=["audio_file", "text", "speaker_name"],
      )

      def add_features(example):
          audio_path = os.path.join(final_dataset_path, example['audio_file'])
          example['audio'] = audio_path

          speaker_name = example["speaker_id"]
          example['gender'] = "unknown"
          if speaker_name.startswith("F_"):
            example['gender'] = "female"
          elif speaker_name.startswith("M_"):
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
      
      dataset_dict.push_to_hub(args.dataset_name)
      print("Dataset pushed to Hugging Face Hub successfully!")

    except FileNotFoundError as e:
      print(f"Error: {e}")
      print("Please ensure 'metadata_train.csv', 'metadata_test.csv', and the 'wavs' folder are correctly placed inside the output directory.")
    except Exception as e:
      print(f"An unexpected error occurred during loading or uploading: {e}")