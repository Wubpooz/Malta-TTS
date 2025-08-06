**NOTE** For those interested in highly detailed topics such as advanced training parameters, memory management, optimizing audio data quality, and understanding model performance metrics, Understanding Model Training Results, please refer to the [XTTS Model Finetuning Guide (Advanced Version)](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Advanced-Version)). Additionally, please refer to Coqui's documentation & forums for anything not covered in these guides https://docs.coqui.ai/en/latest/index.html

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
   - [Requirements](#requirements)
   - [Step-by-Step Process](#step-by-step-process)
   - [Common Issues](#common-issues)
2. [Core Concepts](#core-concepts)
   - [Understanding Finetuning](#understanding-finetuning)
   - [Preparing Audio Data](#preparing-audio-data)
   - [Training Process Overview](#training-process-overview)
3. [Technical Deep Dives](#technical-deep-dives)
   - [Training Parameters](#training-parameters)
   - [Memory Management](#memory-management)
   - [Advanced Features](#advanced-features)
4. [Troubleshooting & Optimization](#troubleshooting--optimization)
   - [Improving Model Quality](#improving-model-quality)
   - [Performance Optimization](#performance-optimization)
5. [Deploying Your Model](#deploying-your-model)

---

## 1. Quick Start Guide


### Requirements

* **GPU Hardware**: 
  - Windows: NVIDIA GPU with 12GB+ VRAM (8GB possible but requires 24GB+ system RAM)
  - Linux: NVIDIA GPU with 16GB+ VRAM recommended
* **System RAM**: 16GB absolute minimum, 24GB+ recommended
* **Storage**: 21GB free space (used for temporary work area & model storage)
* **Audio Data**: 2-3 minutes minimum of clean voice samples (5+ minutes recommended)
* **Internet**: Required for downloading Whisper transcription model

### Initial Setup

1. Install and launch AllTalk
2. Download at least one XTTS model:
   - Open AllTalk's main interface
   - Navigate to: `TTS Engine Settings` > `XTTS TTS` > `Model/Voices Download`
   - Download XTTS v2.0.3 model (or v2.0.3)
3. Close any GPU-intensive applications (especially important with 10GB or less VRAM)

### Audio Preparation

Place your audio files in `/finetune/put-voice-samples-in-here/`

**Tips for good results:**
- Use clear recordings without background noise or music
- Mix of different sentence lengths
- Natural speaking pace
- Consistent audio volume
- Supported formats: MP3, WAV, or FLAC
- The system will automatically split long audio files into smaller segments
- Ensure your audio only contains the target speaker's voice
- Remove any sections containing other speakers before training
- Clean, single-speaker audio produces the best results

## Launch Finetuning

```bash
# For standalone AllTalk:
cd alltalk_tts
./start_finetune.sh    # (Linux)
start_finetune.bat     # (Windows)
```

## Step-by-Step Process

### Step 1: Dataset Creation
**Remember:** Each tab in the interface contains comprehensive help

After uploading your audio files.

1. **Basic Settings:**
   - Project Name: Enter a unique name (e.g., speaker's name)
   - Language: Select your audio's language
   
2. **Processing Options:**
   - Whisper Model:
     * `large-v3`: Best quality, slower processing (recommended)
     * `medium`: Faster processing, good quality
     * `small`: Quick testing, lower quality
   
3. **Advanced Settings (defaults usually work best):**
   - Min Audio Length: Controls min segment length (default: 2s)
   - Max Audio Length: Controls max segment length (default: 10s)
   - Evaluation Split: How much data for testing (default: 15%)
   - VAD: Voice detection (typically leave this enabled)
   - BPE Tokenizer: Custom tokenizer for unique speech patterns or languages
     * Enable for non-English or unique speaking styles
     * Disabled by default - only needed for special cases
   - Model Precision: 
     * Mixed: Best for modern GPUs (default)
     * FP32: For older GPUs
     * FP16: Maximum efficiency, newer GPUs only

4. Click "Step 1 - Create dataset"

5. **Optional but Recommended:**
   - Go to Dataset Validation tab
   - Click "Run Validation"
   - Review and correct any transcription errors
   - Better transcriptions = better training results

### Step 2: Training

1. **Basic Settings:**
   - Project Name: Same as Step 1
   - Model Selection: Choose your base XTTS model or model you want to further train

2. **Training Parameters:**
   - Epochs: How many training cycles
     * Start with 10 (default)
     * More epochs = better results but longer training
   - Batch Size: Samples processed at once
     * Start with 4 (default)
     * Reduce to 2 if out of memory
   - Learning Rate: How fast the model learns
     * Start with 5e-6 (default)
     * Lower = more stable but slower
   - Optimizer: How the model updates
     * AdamW (default) works best

3. Click "Step 2 - Run the training"

4. **Monitor Progress:**
   - Watch the graphs for training progress
   - Loss values should generally decrease
   - Check the estimated completion time
   - Built-in help explains all graphs in detail and the wiki contains more information

### Step 3: Testing

1. Click "Refresh Dropdowns" to load available files

2. **Test Settings:**
   - Select your trained model files
   - Choose reference audio (6+ seconds long, but you can change this)
   - Select language
   - Enter test text

3. Click "Generate TTS" to test

4. **Testing Tips:**
   - Try different reference files
   - Test with various text lengths
   - Check pronunciation and tone
   - If quality is poor, try training longer

### Step 4: Export

1. **Export Settings:**
   - Click "Refresh Dropdowns"
   - Choose a folder name for your model
   - Select overwrite options

2. **Export Process:**
   - Click "Compact and move model"
   - Model exports to `/models/xtts/{your-folder-name}/`
   - Voice samples organized by length in `wavs` folder
   - Check `audio_report.txt` for sample guidance

3. **Cleanup:**
   - After successful export, use "Delete generated training data"
   - Optionally delete original voice samples

### Using Your Model

1. Find your exported model in `/models/xtts/{your-folder-name}/`
2. Select voice samples from the `wavs/suitable` folder
3. Copy preferred samples to AllTalk's 'voices' folder
4. Use in main AllTalk interface

### Common Issues

- **Out of Memory?** 
  - Reduce batch size to 2
  - Enable gradient accumulation
  - Close other applications

- **Poor Quality?**
  - Check audio sample quality
  - Try more training epochs
  - Validate transcriptions
  - Try different reference files

- **Need More Help?**
  - Check built-in documentation in each tab
  - Reference the Finetune [XTTS Model Finetuning Guide (Advanced Version)](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Advanced-Version))

Remember: Each tab in the interface contains comprehensive help - click the dropdown arrows to access detailed information about every setting and process.

---

## 2. Core Concepts

### Understanding Finetuning
Finetuning is the process of adjusting a pre-trained model using new, specific voice data to better capture unique vocal characteristics. This is useful for creating personalized voice models that retain the base model's general abilities while improving on the nuances of your specific dataset.

**When to Finetune**:
- Base model doesn’t capture desired vocal qualities.
- Adaptation needed for accents, unique voices, or specific speaking styles.
- Custom voices for particular applications.

### Preparing Audio Data
- **Duration**: Minimum of 3 minutes, 5–10 minutes recommended.
- **Content**: Clear, noise-free, and consistent volume. Avoid background sounds or music.
- **Format**: MP3, WAV, or FLAC. Any sample rate works, and stereo or mono is supported.

**Tips**:
- Use varied sentence lengths, tones, and speaking speeds.
- Pre-process to remove background noise, split long files if needed, and ensure content quality.

### Training Process Overview
1. **Dataset Generation**:
   - Audio is segmented and transcribed using Whisper.
   - Dataset split into training and evaluation sets (`metadata_train.csv` and `metadata_eval.csv`).
   - Use the Dataset Validation tab to look for inconsistencies in the transcriptions & correct as necessary.

2. **Model Training**:
   - Set training parameters and monitor progress. Model learns to emulate the target voice.
   - **Validation Paths**: Paths for validation data and Whisper model transcription are used to validate and monitor quality. Customize these paths if you want to specify a different dataset for validation.

3. **Testing**:
   - Run tests with different text inputs to evaluate quality, pronunciation, and emotional range.

4. **Model Export**:
   - Compact model, save essential files, and clean up training artifacts.
   - After the model is compacted and moved, select the wav files from the models folder for using with voice cloning [Details here](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Simple-Version)#5-deploying-your-model).

---

## 3. Technical Deep Dives

### Training Parameters

#### Epochs
An **epoch** is a full pass through the dataset. Choose the number of epochs based on the desired outcome:
- Standard voices: 10–20 epochs
- Highly unique voices or accents: 40+ epochs
- Complex voices or new languages: May require 100+ epochs.

**Tip**: Monitor loss values to avoid overfitting. See [Overtraining](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Advanced-Version)#understanding-model-training-results).

#### Batch Size
- **Small Batch Size** (4–8): Lower memory use, more frequent updates.
- **Large Batch Size** (32–64): Faster processing, requires more memory.

**Gradient Accumulation**: If VRAM is limited, simulate larger batch sizes with gradient accumulation:
   ```python
   batch_size = 8
   gradient_accumulation_steps = 4  # Effective batch size = 32
   ```

#### Learning Rate
Controls how quickly the model learns:
- 1e-6 to 5e-6: Stable, slow learning.
- 1e-5: Balanced, suitable for most cases.
- 1e-3 and higher: Fast but can be unstable.

**Schedulers**: Use learning rate scheduling (e.g., cosine annealing or exponential decay) for better results over long training runs.

### Memory Management
- **Windows**: Can use system RAM as extended VRAM.
- **Linux**: Limited to physical VRAM, requiring higher VRAM for reliable operation.

**Optimization Strategies**:
1. Lower batch size or increase gradient accumulation if memory is limited.
2. Adjust worker threads and limit audio length for efficiency.
3. **Signal Handling**: Use the GUI’s stop option or standard interrupts to safely halt training without corrupting model state.

### Advanced Features

#### BPE Tokenization
The script uses **Byte-Pair Encoding (BPE)** tokenization, which helps the model handle complex text, diverse languages, accents, and dialects more effectively. This feature allows the model to better manage unique vocabularies and speech patterns.

---

## 4. Troubleshooting & Optimization

### Improving Model Quality

You can read advanced sections on this topic [here](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Advanced-Version)#understanding-model-training-results)

1. **Signs of Good Training Progression**:
   - Loss values consistently decrease over epochs. Example:
     ```
     Text Loss: 0.02 -> 0.015 -> 0.012
     Mel Loss: 4.5 -> 3.8 -> 3.2
     Average Loss: 4.0 -> 3.5 -> 3.0
     ```
   - Model saves as "BEST MODEL" at new performance milestones.

2. **Recognizing Overtraining**:
   - Loss values plateau or start increasing, e.g.,:
     ```
     Epoch 5: Text Loss = 0.009, Mel Loss = 2.9
     Epoch 7: Text Loss = 0.010, Mel Loss = 3.1  # Performance worsening
     ```
   - Solutions: Implement early stopping, reduce training epochs, or lower learning rate.

### Performance Optimization

1. **Hardware**:
   - Adjust batch size based on GPU capability.
   - Optimize worker count and monitor VRAM.

2. **Configuration**:
   - Use efficient settings, e.g., float16 precision if supported.
   - Regularly monitor memory and processing efficiency.
   - Adjust `gradient_accumulation_steps` for limited VRAM.

---

## 5. Deploying Your Model

### Model Export and Storage

After finetuning, export and organize the model files:
- **Essential Files**:
   - `model.pth`: Main model
   - `config.json`: Configuration settings
   - `vocab.json`: Tokenizer vocabulary
   - `speakers_xtts.pth`: Speaker embeddings
   - `dvae.pth`: Discrete Variational Autoencoder (DVAE) model file
   - `mel_stats.pth`: Mel spectrogram statistics

- **Storage Requirements**:
   - Model size: ~1.5GB
   - Reference audio varies based on content.

In the `wavs` folder, under the models folder, you will find an `audio_report.txt` along with a few folders. The `audio_report.txt` will explain which wav files are suitable to use for voice cloning with the XTTS model you have finetuned, along with details of what you can do with the other files if you so need. The files you want to use for voice cloning should be copied into your `alltalk_tts\voices\` folder.

### Integration and Usage

**Folder Structure**:
```
models/
└── xtts/
    └── your_model_name/
        ├── wavs/ 
        │    ├── audio_report.txt (READ THIS FILE)
        │    ├── suitable/ 
        │    ├── too_long/ 
        │    └── too_short/ 
        ├── model.pth
        ├── config.json
        ├── vocab.json
        ├── dvae.pth
        ├── mel_stats.pth
        └── speakers_xtts.pth
```