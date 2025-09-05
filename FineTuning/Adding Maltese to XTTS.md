# Adding Maltese to XTTS
## Table of Contents
- [Adding Maltese to XTTS](#adding-maltese-to-xtts)
  - [Table of Contents](#table-of-contents)
  - [XTTS Model](#xtts-model)
    - [Core Components](#core-components)
    - [Features](#features)
    - [License](#license)
    - [Inference](#inference)
    - [Manual Inference](#manual-inference)
        - [inference parameters](#inference-parameters)
        - [Inference](#inference-1)
    - [Training](#training)
      - [Run demo on Colab](#run-demo-on-colab)
      - [Advanced training](#advanced-training)
    - [Byte-Pair Encoding (BPE) Tokenization](#byte-pair-encoding-bpe-tokenization)
  - [Finetuning](#finetuning)
    - [Creating a new Dataset](#creating-a-new-dataset)
  - [Dataset Requirements](#dataset-requirements)
    - [Minimum Requirements](#minimum-requirements)
    - [Maltese data](#maltese-data)
    - [Data Augmentations](#data-augmentations)
    - [Using an existing Dataset](#using-an-existing-dataset)
      - [Multiple Voice Training](#multiple-voice-training)
      - [HyperParameters](#hyperparameters)
      - [Essential Files](#essential-files)
      - [Notes](#notes)
      - [Example of configuration](#example-of-configuration)
      - [Strategies for Managing Minima](#strategies-for-managing-minima)
    - [Training](#training-1)
      - [Forgetting Mitigation Strategies](#forgetting-mitigation-strategies)
        - [LoRA (Low-Rank Adaptation) - RECOMMENDED](#lora-low-rank-adaptation---recommended)
        - [Layer Freezing](#layer-freezing)
        - [Standard Fine-tuning](#standard-fine-tuning)
      - [Monitoring Training](#monitoring-training)
    - [Inference](#inference-2)
    - [Inference with Speaker Embeddings](#inference-with-speaker-embeddings)
  - [Evaluation \& Testing](#evaluation--testing)
    - [1. Objective Metrics](#1-objective-metrics)
    - [2. Subjective Evaluation](#2-subjective-evaluation)
    - [3. Speaker Similarity](#3-speaker-similarity)
  - [Common Issues \& Solutions](#common-issues--solutions)
    - [Catastrophic Forgetting](#catastrophic-forgetting)
    - [Poor Maltese Pronunciation](#poor-maltese-pronunciation)
    - [Code-Switching Failures](#code-switching-failures)
    - [OOM Errors](#oom-errors)
  - [Best Practices](#best-practices)
  - [Future Improvements](#future-improvements)
  - [References](#references)

## XTTS Model
ⓍTTS is a Text-to-Speech model that lets you clone voices in different languages by using just a quick 3-second audio clip. Built on the 🐢Tortoise, ⓍTTS has important model changes that make cross-language voice cloning and multi-lingual speech generation super easy. There is no need for an excessive amount of training data that spans countless hours.  
This is the same model that powers [Coqui Studio](https://coqui.ai/), and [Coqui API](https://docs.coqui.ai/docs), however we apply a few tricks to make it faster and support streaming inference.  

### Core Components
1. **GPT Model**: Generates semantic tokens from text
2. **DVAE (Discrete VAE)**: Encodes/decodes audio features
3. **HiFiGAN Decoder**: Converts features to waveform
4. **BPE Tokenizer**: Handles text tokenization

### Features
- Voice cloning.
- Cross-language voice cloning.
- Multi-lingual speech generation.
- 24khz sampling rate.
- Streaming inference with < 200ms latency. (See [Streaming inference](#streaming-inference))
- Fine-tuning support. (See [Training](#training))
- Support for 16 languages: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu) and Korean (ko).

### License
This model is licensed under [Coqui Public Model License](https://coqui.ai/cpml). Note that the Coqui company does no longer exists. Support is available at [Coqui-TTS](https://github.com/idiap/coqui-ai-TTS).


&nbsp;  
### Inference
```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav=["/path/to/target/speaker.wav"], 
                # Or ["/path/to/target/speaker.wav", "/path/to/target/speaker_2.wav", "/path/to/target/speaker_3.wav"] for multiple references
                # Or "target_speaker_name" for Coqui speakers
                language="en",
                split_sentences=True
                )
```
You can optionally disable sentence splitting for better coherence but more VRAM and possibly hitting models context length limit.  


&nbsp;  
### Manual Inference
If you want to be able to `load_checkpoint` with `use_deepspeed=True` and enjoy the speedup, you need to install deepspeed first.  
```console
pip install deepspeed==0.10.3
```

&nbsp;  
##### inference parameters
- `text`: The text to be synthesized.
- `language`: The language of the text to be synthesized.
- `gpt_cond_latent`: The latent vector you get with get_conditioning_latents. (You can cache for faster inference with same speaker)
- `speaker_embedding`: The speaker embedding you get with get_conditioning_latents. (You can cache for faster inference with same speaker)
- `temperature`: The softmax temperature of the autoregressive model. Defaults to 0.65.
- `length_penalty`: A length penalty applied to the autoregressive decoder. Higher settings causes the model to produce more terse outputs. Defaults to 1.0.
- `repetition_penalty`: A penalty that prevents the autoregressive decoder from repeating itself during decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc. Defaults to 2.0.
- `top_k`: Lower values mean the decoder produces more "likely" (aka boring) outputs. Defaults to 50.
- `top_p`: Lower values mean the decoder produces more "likely" (aka boring) outputs. Defaults to 0.8.
- `speed`: The speed rate of the generated audio. Defaults to 1.0. (can produce artifacts if far from 1.0)
- `enable_text_splitting`: Whether to split the text into sentences and generate audio for each sentence. It allows you to have infinite input length but might loose important context between sentences. Defaults to True.


&nbsp;  
##### Inference
```python
import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# use your values
CONFIG_PATH = "/path/to/xtts/config.json"
XTTS_CHECKPOINT = "/path/to/xtts/"
TOKENIZER_PATH = "path/to/xtts/vocab.json"
SPEAKER_REFERENCE = "reference.wav"
OUTPUT_WAV_PATH = "xtts.wav"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

print("Inference...")
out = model.inference(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)

# Use this commented chunk for streaming
# wav_chuncks = []
# for i, chunk in enumerate(chunks):
#     if i == 0:
#         print(f"Time to first chunck: {time.time() - t0}")
#     print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
#     wav_chuncks.append(chunk)
# wav = torch.cat(wav_chuncks, dim=0)

torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)
```



&nbsp;  
### Training
#### Run demo on Colab
The Colab Notebook is available [here](https://colab.research.google.com/drive/1GiI4_X724M8q2W-zZ-jXo7cWTV7RfaH-?usp=sharing).  

1. Open the Colab notebook and start the demo by runining the first two cells (ignore pip install errors in the first one).
2. Click on the link "Running on public URL:" on the second cell output.
3. On the first Tab (1 - Data processing) you need to select the audio file or files, wait for upload, and then click on the button "Step 1 - Create dataset" and then wait until the dataset processing is done.
4. Soon as the dataset processing is done you need to go to the second Tab (2 - Fine-tuning XTTS Encoder) and press the button "Step 2 - Run the training" and then wait until the training is finished. Note that it can take up to 40 minutes.
5. Soon the training is done you can go to the third Tab (3 - Inference) and then click on the button "Step 3 - Load Fine-tuned XTTS model" and wait until the fine-tuned model is loaded. Then you can do the inference on the model by clicking on the button "Step 4 - Inference".

To learn how to use this Colab Notebook please check the [tutorial video](https://www.youtube.com/watch?v=8tpDiiouGxc&feature=youtu.be).  


&nbsp;  
#### Advanced training
A recipe for `XTTS_v2` GPT encoder training using `LJSpeech` dataset is [available](https://github.com/coqui-ai/TTS/tree/main/recipes/ljspeech/xtts_v1/train_gpt_xtts.py).  

You need to change the fields of the `BaseDatasetConfig` to match your dataset and then update `GPTArgs` and `GPTTrainerConfig` fields as you need. By default, it will use the same parameters that XTTS v1.1 model was trained with. To speed up the model convergence, as default, it will also download the XTTS v1.1 checkpoint and load it.  

After training, run inference as described in the [Inference](#inference) section.  


&nbsp;  
### Byte-Pair Encoding (BPE) Tokenization
Byte-Pair Encoding (BPE) is a tokenization technique that breaks text into subword units, making it easier for the model to handle unique or complex words, accents, and phonetic variations.
Benefits of BPE Tokenization:
- Handles Unknown Words: Breaks down uncommon words, allowing the model to learn parts of words it hasn’t encountered before.
- Improves Accuracy with Accents and Dialects: By encoding subword units, the model can better capture pronunciation nuances found in accents or dialects.
- Supports Multi-Language Training: BPE helps the model generalize across languages by creating shared subword representations.
XTTS Implementation:
   - XTTS models use a specialized tokenizer with support for multiple languages and unique speech patterns.
   - The script also includes options to customize token frequency thresholds, vocabulary size, and stop tokens.


[Source](https://github.com/coqui-ai/TTS/tree/main/docs/source/models/xtts.md)


---


&nbsp;  
&nbsp;  
## Finetuning
### Creating a new Dataset
1) Record Your Audio: Find a native Maltese speaker with a clear voice. Record them reading Maltese sentences in a quiet environment with a decent microphone. Aim for **at least 1 hour of audio** Export the audio as a single `.wav` file, with a sample rate of **22050 Hz** and in **mono**. Quality over quantity.
2) Transcribe and segment the audio: Use a transcription tool that provides per-word timestamps. A great open-source option is `faster-whisper`. Output the text in the LJSpeech format (wav folder and `metadata.csv` file in the format `filename|transcription|normalized_transcription`).
3) Split the Dataset: Split your `metadata.csv` into two files: `metadata_train.csv` (for training, ~95% of the lines) and `metadata_eval.csv` (for validation, ~5% of the lines).



&nbsp;  
&nbsp;  
## Dataset Requirements
### Minimum Requirements
- **Duration**: 1+ hours of clean audio (10+ hours recommended)
- **Quality**: Studio or high-quality recordings
- **Sample Rate**: 22050 Hz (will be resampled if different)
- **Format**: WAV, mono channel
- **Transcription**: Accurate, normalized text

&nbsp;  
### Maltese data
- [MASRI dev Dataset](https://huggingface.co/datasets/MLRS/masri_dev) - 1h of Maltese audio data
- [MASRI HEADSET](https://huggingface.co/datasets/Bluefir/MASRI_HEADSET_Corpus_v2) - 6h of Maltese audio data, multiple speakers
- [Maltese Common Voice](https://huggingface.co/datasets/common_voice/mt) - 1,000+ hours of Maltese audio data
- [Huge text Corpus](https://mlrs.research.um.edu.mt/index.php?page=corpora) and [HuggingFace](https://huggingface.co/datasets/MLRS/korpus_malti)
- [Maltese tokenizer](https://github.com/UMSpeech/MASRI/blob/main/masri/tokenise/tokenise.py)
- [Sentimental Maltese data](https://github.com/jerbarnes/typology_of_crosslingual/tree/master/data/sentiment/mt)
- 7,000h of unlabeled data (=> use Granary pipeline)


&nbsp;  
### Data Augmentations
- Speed perturb (0.9×/1.0×/1.1×), room IRs, light noise, pitch shift ≤ ±100 cents (sparingly), SpecAugment on spectrograms for the acoustic model.
- Add code-switch examples (Maltese + English/Italian) — realistic for Malta and improves robustness.
- Multispeaker Maltese (+ neighbors) for robustness: combine Maltese with Italian/Arabic/English; ensure at least ~30–60 min per speaker for stability.


&nbsp;  
### Using an existing Dataset
1) Setup your dataset
   1) [Check the quality of your dataset](https://github.com/coqui-ai/TTS/tree/main/docs/source/what_makes_a_good_dataset.md)
      - Gaussian like distribution on clip and text lengths
      - Mistake free: remove any wrong or broken files, check annotations, compare transcript and audio length.
      - Noise free: background noise might lead your model to struggle, especially for a good alignment.
      - Compatible tone and pitch among voice clips: for instance, if you are using audiobook recordings for your project, it might have impersonations for different characters in the book. These differences between samples downgrade the model performance.
      - Good phoneme coverage.
      - Naturalness of recordings.
      - Quantization level of the clips: if your dataset has a very high bit-rate, that might cause slow data-load time and consequently slow training. It is better to reduce the sample-rate of your dataset to around 16000-22050.

      There are 2 notebooks to help you determine the quality of the dataset: [CheckSpectrograms](https://github.com/coqui-ai/TTS/tree/main/docs/notebooks/CheckSpectrograms.ipynb) and [AnalyzeDataset](https://github.com/coqui-ai/TTS/tree/main/docs/notebooks/AnalyzeDataset.ipynb). The first one measures the noise level and find good audio processing parameters. The second one checks the distribution of clip and text lengths.

   2) [Format your dataset](https://github.com/coqui-ai/TTS/tree/main/docs/source/formatting_dataset.md)
      - The speech must be divided into audio clips and each clip needs transcription. It is important to use a lossless audio file format to prevent compression artifacts. We recommend using `wav` file format.  
      - We recommend the following format delimited by `|`. In the following example, `audio1`, `audio2` refer to files `audio1.wav`, `audio2.wav` etc.
        ```
        # metadata.txt

        audio1|This is my sentence.|This is my sentence.
        audio2|1469 and 1470|fourteen sixty-nine and fourteen seventy
        audio3|It'll be $16 sir.|It'll be sixteen dollars sir.
        ...
        ```
        *If you don't have normalized transcriptions, you can use the same transcription for both columns. If it's your case, we recommend to use normalization later in the pipeline, either in the text cleaner or in the phonemizer.*
      
      - The dataset structure should be the following:
        ```
        /MyTTSDataset
          |
          | -> metadata.txt
          | -> /wavs
            | -> audio1.wav
            | -> audio2.wav
            | ...
        ```
        This is taken from the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset.

     3) Using Your Dataset
        After you collect and format your dataset, you need to check two things. Whether you need a `formatter` and a `text_cleaner`. The `formatter` loads the text file (created above) as a list and the `text_cleaner` performs a sequence of text normalization operations that converts the raw text into the spoken representation (e.g. converting numbers to text, acronyms, and symbols to the spoken format):  
          - If you use a different dataset format than the LJSpeech or the other public datasets that 🐸TTS supports, then you need to write your own `formatter`.  
            What you get out of a `formatter` is a `List[Dict]` in the following format:
            ```
            >>> formatter(metafile_path)
            [
              {"audio_file":"audio1.wav", "text":"This is my sentence.", "speaker_name":"MyDataset", "language": "lang_code"},
              {"audio_file":"audio1.wav", "text":"This is maybe a sentence.", "speaker_name":"MyDataset", "language": "lang_code"},
              ...
            ]
            ```
            Each sub-list is parsed as ```{"<filename>", "<transcription>", "<speaker_name">]```. ```<speaker_name>``` is the dataset name for single speaker datasets and it is mainly used in the multi-speaker models to map the speaker of the each sample. But for now, we only focus on single speaker datasets.  

            The purpose of a `formatter` is to parse your manifest file and load the audio file paths and transcriptions.  
            Then, the output is passed to the `Dataset`. It computes features from the audio signals, calls text normalization routines, and converts raw text to
            phonemes if needed.  
          - If your dataset is in a new language or it needs special normalization steps, then you need a new `text_cleaner`.

    4) Loading your dataset
        ```python
        from TTS.tts.datasets import load_tts_samples

        # custom formatter implementation
        def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
          """Assumes each line as ```<filename>|<transcription>```
          """
          txt_file = os.path.join(root_path, manifest_file)
          items = []
          speaker_name = "my_speaker"
          with open(txt_file, "r", encoding="utf-8") as ttf:
            for line in ttf:
              cols = line.split("|")
              wav_file = os.path.join(root_path, "wavs", cols[0])
              text = cols[1]
              items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
          return items

        train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=formatter)
        ```

        See `TTS.tts.datasets.TTSDataset`, a generic `Dataset` implementation for the `tts` models.  
        See `TTS.vocoder.datasets.*`, for different `Dataset` implementations for the `vocoder` models.  
        See `TTS.utils.audio.AudioProcessor` that includes all the audio processing and feature extraction functions used in a `Dataset` implementation. Feel free to add things as you need.  

2) Download the xtts_v2 model: `tts --model_name tts_models/multilingual/multi-dataset/xtts_v2`

3) Setup the model config for fine-tuning:
    - Edit the fields in the ```config.json``` file if you want to use ```TTS/bin/train_tts.py``` to train the model.
    - Edit the fields in one of the training scripts in the ```recipes``` directory if you want to use python.
    - Use the command-line arguments to override the fields like ```--coqpit.lr 0.00001``` to change the learning rate.
     Some of the important fields are `datasets`, `run_name`, `output_path`, `lr`, and `audio` (audio characteristics).

4) Start fine-tuning, use restore_path to specify the path to the pre-trained model, here are the commands for the previous cases respectively:
    - ```bash
      CUDA_VISIBLE_DEVICES="0" python recipes/ljspeech/glow_tts/train_glowtts.py \
          --restore_path  /home/ubuntu/.local/share/tts/tts_models--en--ljspeech--glow-tts/model_file.pth
      ```
    - ```bash
      CUDA_VISIBLE_DEVICES="0" python TTS/bin/train_tts.py \
          --config_path  /home/ubuntu/.local/share/tts/tts_models--en--ljspeech--glow-tts/config.json \
          --restore_path  /home/ubuntu/.local/share/tts/tts_models--en--ljspeech--glow-tts/model_file.pth
      ```
    - ```bash
      CUDA_VISIBLE_DEVICES="0" python recipes/ljspeech/glow_tts/train_glowtts.py \
          --restore_path  /home/ubuntu/.local/share/tts/tts_models--en--ljspeech--glow-tts/model_file.pth
          --coqpit.run_name "glow-tts-finetune" \
          --coqpit.lr 0.00001
      ```

[Source](https://github.com/coqui-ai/TTS/tree/main/docs/source/finetuning.md)


&nbsp;  
#### Multiple Voice Training
1. Initial Voice Training:
  - Start with a base model and finetune it on the first voice.
  - Export the trained model as a checkpoint.
2. Additional Voices:
  - Use the checkpoint from the previous training as a base.
  - Finetune the model with the new voice data, gradually adapting it to recognize multiple speakers.
3. Balancing Training Time:
  - Divide training time across voices for even quality.
  - Monitor each voice to ensure the model maintains quality for all trained voices.


&nbsp;  
#### HyperParameters
   - Epochs: How many training cycles, start with 10 (more epochs = better results but longer training).
   - Batch Size: Samples processed at once, start with 4 and reduce to 2 if out of memory but can be increased to 32 or 64.
     - If VRAM is limited, use gradient accumulation to simulate larger batch sizes (batch_size = 8, gradient_accumulation_steps = 4 => effective batch size = 32)
   - Learning Rate: How fast the model learns, start with 5e-6 (lower = more stable but slower)
   - Optimizer: How the model updates, AdamW (default) works best ([Should I try multiple optimizers when fine-tuning a pre-trained Transformer for NLP tasks? Should I tune their hyperparameters?](https://aclanthology.org/2024.eacl-long.157/))
   - Scheduler: How the learning rate changes, CosineAnnealingWarmRestarts works best:
     - Cosine Annealing with Warm Restarts (reduces the learning rate in cycles to explore different minima, ideal for long training runs where you want the model to avoid settling on a local minimum):
        ```
        Learning Rate
             ^
        η_max|   /\    /\    /\
             |  /  \  /  \  /  \
             | /    \/    \/    \
        η_min|_____________________> Time
        ```
    - Step Decay
    - Exponential Decay


&nbsp;  
#### Essential Files
model.pth: Main model  
config.json: Configuration settings  
vocab.json: Tokenizer vocabulary  
speakers_xtts.pth: Speaker embeddings  
dvae.pth: Discrete Variational Autoencoder (DVAE) model file  
mel_stats.pth: Mel spectrogram statistics  


&nbsp;  
#### Notes
- Requirements: GPU with 16GB+ VRAM and 24GB+ RAM, 30G+ free storage.
- Use a small learning rate to avoid overfitting and forgetting the training data.
  - 1e-6 to 5e-6: Stable, slow learning.
  - 1e-5: Balanced, suitable for most cases.
  - 1e-3 and higher: Fast but can be unstable.
- Monitoring progress: loss values should generally decrease consistently over epochs.
- Regular Monitoring: Use tools like `nvidia-smi` (for NVIDIA GPUs) to monitor GPU memory usage in real time. This helps identify if adjustments are needed mid-training.
- Overtraining: Loss values plateau or start increasing, indicating overtraining => reduce learning rate, epochs or early stopping.
- Check for GPU Memory Fragmentation: Restart training if you notice fragmentation from long sessions, as this can sometimes free up memory.
- Garbage Collection in PyTorch: utilizes Python’s `gc` (garbage collection) module to manage memory.


&nbsp;  
#### Example of configuration
- 1000 epoch for training (https://github.com/erew123/alltalk_tts?#-finetuning-a-model)
- 396 samples on V202, 44 epochs, batch size of 7, grads 10, with a max sample length of 11 seconds.
- Or 11 epochs, 6 sample, 6 grads, 11 seconds and it was fine. I had 89 samples.


&nbsp;  
#### Strategies for Managing Minima
- Batch Size:
  - Small Batch Sizes (4–8): Introduce more randomness into the training process, which can help the model escape poor local minima. Smaller batches, however, may increase training noise and slow down convergence.
  - Large Batch Sizes (32–64): Provide stability, helping the model converge more predictably. Larger batches are less likely to "escape" local minima, but they can yield sharper minima, which may lead to overfitting.
- Learning Rate:
  - High Learning Rate: Speeds up training but can cause the model to "jump over" good minima, leading to instability or convergence in a suboptimal local minimum.
  - Low Learning Rate: Helps the model make finer adjustments and settle in more favorable minima. It can be beneficial to start with a high learning rate and decrease it over time, often through a scheduler.
- Gradient Accumulation and Gradient Clipping:
  - Gradient Accumulation: Accumulates gradients over multiple smaller batches, simulating a larger batch size while still allowing variability. This helps the model balance between escaping poor minima and achieving stable convergence.
  - Gradient Clipping: Limits the size of gradients to prevent large, erratic updates. This can help stabilize training when the model is near a minimum, reducing the chance of "overshooting" it.
- Early Stopping: Set a condition to stop training when the model stops improving for a set number of epochs. Early stopping can prevent the model from descending too far into overfitting, keeping it in a minimum that generalizes better.
- Warm Restarts:
  A technique that resets the learning rate at intervals, allowing the model to re-explore the loss landscape:
  - Warm restarts encourage the model to "escape" poor minima by periodically increasing the learning rate, then lowering it again as training continues.
  - Useful in longer training runs to avoid the model settling in sharp, potentially overfitted minima.



&nbsp;  
&nbsp;  
### Training
```bash
# CUDA setup for optimal performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# Enable TF32 for Ampere GPUs
export TORCH_ALLOW_TF32=1
```
&nbsp;  
```bash
python train_gpt_xtts.py \
--output_path ./xtts_maltese_model/ \
--metadatas path/to/your/maltese_dataset/metadata_train.csv,path/to/your/maltese_dataset/metadata_eval.csv,mt \
--num_epochs 10 \
--batch_size 4 \
--grad_acumm 8 \
--max_text_length 200 \
--max_audio_length 255995 \
--weight_decay 1e-4 \
--lr 5e-6 \
--save_step 5000
```

`--metadatas`: A comma-separated string: `train_metadata_path,eval_metadata_path,language_code`. For Maltese, the ISO 639-1 code is `mt`.  
`--num_epochs`: The number of times to go through the training data. Start with 10 and you can increase it later.  
`--batch_size` & `--grad_acumm`: Control memory usage. If you run out of VRAM, decrease the `batch_size` and increase `grad_acumm` to compensate. An effective batch size is `batch_size * grad_acumm`.  

#### Forgetting Mitigation Strategies
##### LoRA (Low-Rank Adaptation) - RECOMMENDED
```python
lora_config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=16,          # Scaling factor
    target_modules=["c_attn", "c_proj"],  # Target layers
    lora_dropout=0.05,
    bias="none"
)
```
- Preserves original model weights
- Adds trainable adapter layers
- ~10% of parameters trainable

##### Layer Freezing
```python
# Only train final layers
trainable_layers = [
    "text_embedding",  # Vocabulary embeddings
    "layers.29",       # Last 3 transformer layers
    "layers.28",
    "layers.27",
    "final_norm",
    "lm_head"
]
```
- Most conservative approach
- ~15% of parameters trainable
- Slower adaptation

##### Standard Fine-tuning
- All parameters trainable
- Highest risk of catastrophic forgetting
- Use only with mixed language datasets


&nbsp;  
#### Monitoring Training
```python
# Check training metrics
tensorboard --logdir checkpoints/training/logs

# Key metrics to monitor:
- loss/train_loss: Should decrease steadily
- loss/eval_loss: Should decrease, watch for overfitting
- grad_norm: Should remain stable (<10)
- learning_rate: Check scheduler working
```


&nbsp;  
&nbsp;  
### Inference
```python
from TTS.api import TTS
import torch

config_path = "xtts_maltese_finetuned/config.json"
model_path = "xtts_maltese_finetuned/best_model.pth" # Or the latest checkpoint
speakers_json_path = "xtts_maltese_finetuned/speakers.json" # If you have speaker information
vocab_path = "xtts_maltese_finetuned/vocab.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

maltese_text = "Bonġu, kif int?"
reference_audio = "path/to/maltese/speaker.wav"

tts = TTS(model_path=model_path, config_path=config_path, progress_bar=True).to(device)

tts.tts_to_file(text=maltese_text, speaker_wav=reference_audio, language="mt", file_path="output_maltese.wav")
```



&nbsp;  
### Inference with Speaker Embeddings
```python
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torchaudio

model_output_path = "./xtts_maltese_model/run/training/..."
xtts_checkpoint = os.path.join(model_output_path, "best_model.pth")
xtts_config = os.path.join(model_output_path, "config.json")
xtts_vocab = "./xtts_maltese_model/ready/vocab.json"
speaker_audio_file = "path/to/your/maltese_speaker_reference.wav" # A clean, 10-15 second audio clip of your Maltese speaker.

print("Loading model...")
config = XttsConfig()
config.load_json(xtts_config)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Model loaded successfully!")


tts_text = "Dan huwa test tal-mudell tat-taħdit il-ġdid tiegħi għal-lingwa Maltija."
lang = "mt"
print(f"Generating speech for: '{tts_text}'")

# Generate conditioning latents from the speaker reference.
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_audio_file)

out = model.inference(
  text=tts_text,
  language=lang,
  gpt_cond_latent=gpt_cond_latent,
  speaker_embedding=speaker_embedding,
  temperature=0.7,  # A higher temperature adds more randomness to the output
  length_penalty=1.0,
  repetition_penalty=10.0,
  top_k=50,
  top_p=0.85
)

torchaudio.save("output_maltese.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
```


---

&nbsp;  
&nbsp;  
## Evaluation & Testing
[Flore+](https://huggingface.co/datasets/openlanguagedata/flores_plus/viewer/mlt_Latn?views%5B%5D=mlt_latn_dev)  

### 1. Objective Metrics
```python
# Character Error Rate (CER) using Whisper or Wisper
from evaluation import calculate_cer

cer = calculate_cer(
    synthesized_audio="output.wav",
    reference_text="Original text",
    whisper_model="large-v3"
)
print(f"CER: {cer:.2%}")  # Target: <5%
```

### 2. Subjective Evaluation
- Automatic Mean Opinion Score (MOS) with native speakers

### 3. Speaker Similarity
```python
# Using speaker verification model
from resemblyzer import VoiceEncoder

encoder = VoiceEncoder()
similarity = encoder.verify(
    reference_audio,
    synthesized_audio
)
print(f"Speaker similarity: {similarity:.2f}")  # Target: >0.8
```



&nbsp;  
&nbsp;  
## Common Issues & Solutions
### Catastrophic Forgetting
Symptoms: Model loses ability to speak other languages  
Solution:  
- Use LoRA or layer freezing
- Include 10–20% of original language data
- Lower learning rate to 1e-6

### Poor Maltese Pronunciation
Symptoms: Incorrect pronunciation of ċ, ġ, għ, ħ, ż  
Solution:  
- Increase Maltese-specific tokens in tokenizer
- Add phoneme-level supervision
- Fine-tune for more epochs

### Code-Switching Failures
Symptoms: Poor quality when mixing languages  
Solution:
- Include code-switched examples in training
- Use language tags: <lang=mt>, <lang=en>
- Train on parallel corpora

### OOM Errors
Symptoms: CUDA out of memory  
Solution:
```bash
# Reduce batch size
--batch_size 2 --grad_acumm 126
# Enable optimizations
--optimizations True --tf32 True
# Use gradient checkpointing
--gradient_checkpointing True
```


&nbsp;  
## Best Practices
1. Data Quality > Quantity
   - 10 hours of clean audio > 100 hours of noisy audio
   - Manual verification of transcriptions
   - Consistent speaker recording conditions
2. Incremental Training
   - Start with 10 epochs, evaluate
   - Continue training if improving
   - Save checkpoints frequently
3. Multi-Stage Training
   ```bash
   # Stage 1: Language adaptation (lower LR)
   --lr 1e-6 --epochs 50

   # Stage 2: Voice fine-tuning (higher LR)
   --lr 5e-6 --epochs 50
   ```
4. Validation Strategy
   - Hold out native speakers for testing
   - Include diverse text (news, conversation, literature)
   - Test edge cases (numbers, dates, foreign words)


&nbsp;  
&nbsp;  
## Future Improvements
1. Use a dataset with a single speaker with 80-90% of maltese speech and 10-20% of the previously learnt languages. The model resizing degrades the english inference already. Claudia Borg agrees, and a fix would be mixed training data (maltese and previously learnt languages). mBERT had 99 empty slots that they could fill in to train so no need to increase the embedding size.
   - Consider implementing consistency loss between different speaker embeddings. Add prosody regularization to maintain natural speech patterns. [Pushing the Limits of Zero-shot End-to-End Speech Translation](https://arxiv.org/abs/2402.10422)
  - Mixed language training (70% Maltese, 30% English/Italian) can improve robustness. Language embeddings should be learned jointly, not separately. [Multilingual Diffusion Models](https://arxiv.org/abs/2412.01271)

2. Feature improvements:
   - Auto-adjust performances based on VRAM:
  ```python
  if torch.cuda.is_available():
    gpu_info = torch.cuda.get_device_properties(0)
    print(f"GPU detected: {gpu_info.name}")
    print(f"   Compute Capability: {gpu_info.major}.{gpu_info.minor}")
    print(f"   Memory: {vram_gb:.1f} GB")
    vram_gb = gpu_info.total_memory / 1024**3
    if vram_gb < 16:
      batch_size = 3
      grad_acumm = 84
    elif vram_gb < 24:
      batch_size = 6
      grad_acumm = 42
    else:
      batch_size = 8
      grad_acumm = 32
    print(f"Batch size: {batch_size}, grad_acumm: {grad_acumm}")
  ```
   - Fix LORA
   - Use ipywidget for Colab inputs
   - Training loss graph
   - Dataset Quality Checks: Add comprehensive validation before training
   - Autodetect dataset formatter in trainGPT based on metadata number of cols and names
   - Early stop by MOS proxy: ASR-CER on TTS–>ASR, and an external prosody score.


1. Phoneme-Level Training
   - Implement Maltese G2P (Grapheme-to-Phoneme)
   - Train on phoneme sequences
   - Better handling of ambiguous pronunciations
2. Emotional TTS
   - See [Rasa Paper](https://www.isca-archive.org/interspeech_2024/srinivasavaradhan24_interspeech.html) for the minimum amount of expressive data needed per emotion, the difficulty of synthesizing certain emotions, and the better expressivity of multi-emotion systems over single-emotion system
   - Collect emotional speech data: [Sentimental Maltese data](https://github.com/jerbarnes/typology_of_crosslingual/tree/master/data/sentiment/mt)
   - Implement emotion embeddings
   - Support for expressive synthesis: `tts.tts_to_file(text="This is a test.", file_path=OUTPUT_PATH, emotion="Happy", speed=1.5)`
3. Lightweight Models
   - Knowledge distillation to smaller models
   - Quantization for mobile deployment
     `python onnx_quantize.py --in xtts_mt.onnx --out xtts_mt_int8.onnx --mode dynamic`
   - ONNX export for cross-platform support:
      `python export_model_onnx.py --checkpoint best_model.pth --out xtts_mt.onnx`
      You may also need to export HiFiGAN separately
   - Packaging for apps
   	-	Android: bundle ONNX models, run with ONNX Runtime (NNAPI/GPU EP if available). Kotlin wrapper; stream PCM to AudioTrack
   	-	iOS: convert to Core ML (coremltools), or use ONNX Runtime Mobile; stream to AVAudioEngine
   	-	Desktop/IoT: Piper or your own C++ runner; one static binary + two model files
   - Pre-warm models on app start (first inference JIT costs)
   - Cache phonemized text and punctuation normalization

4. Auto-split larger audio files into smaller segments
5. GUI (gradio or streamlit) with spectrogram [ExtractTTSSpectrogram](https://github.com/coqui-ai/TTS/tree/main/notebooks/ExtractTTSpectrogram.ipynb)



&nbsp;  
&nbsp;  
## References
- [XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
- [XTTS documentation](https://docs.coqui.ai/en/latest/models/xtts.html)  
- [XTTS demo UI](https://huggingface.co/spaces/coqui/xtts)  
- [Ahnhnh2002's XTTS Finetuning for New Languages](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages%3E)
- [XTTS demo](/TTS/demos/xtts_ft_demo/xtts_demo.py)
- [GPT training recipe](https://github.com/coqui-ai/TTS/tree/dev/recipes/ljspeech/xtts_v1/train_gpt_xtts.py)
- [GPT training recipe enhanced](/FineTuning/train_gpt.py)
- [Google Colab for XTTS Finetuning](https://colab.research.google.com/drive/1GiI4_X724M8q2W-zZ-jXo7cWTV7RfaH-?usp=sharing#scrollTo=Th91ofnQWr8Y)
- [XTTS-webui by daswer123](https://github.com/daswer123/xtts-webui?tab=readme-ov-file) +  [Colab link](https://colab.research.google.com/drive/1MrzAYgANm6u79rCCQQqBSoelYGiJ1qYL#scrollTo=4arC1ywzqfcu)
- [FineTuning documentation for alltalk_tts, based on XTTS](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Simple-Version))
- [Flore+](https://huggingface.co/datasets/openlanguagedata/flores_plus/viewer/mlt_Latn?views%5B%5D=mlt_latn_dev)
- [Should I try multiple optimizers when fine-tuning a pre-trained Transformer for NLP tasks? Should I tune their hyperparameters?](https://aclanthology.org/2024.eacl-long.157/)
- [Example of tokenizer extension](/FineTuning/coqui-ai-TTS-newlanguage.png)  
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)
- [XTTS Paper](https://arxiv.org/abs/2406.04904)
- [Coqui TTS Documentation](https://docs.coqui.ai)
- [MASRI Project](https://mlrs.research.um.edu.mt/)
- [BPE Algorithm](https://arxiv.org/abs/1508.07909)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)


&nbsp;  
**Other models:**
- <https://www.reddit.com/r/mlscaling/comments/1gxakk3/did_a_quick_comparison_of_various_tts_models/>
- <https://github.com/OHF-Voice/piper1-gpl/tree/main>
- <https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a>
