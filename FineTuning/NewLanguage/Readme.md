# XTTS Fine-tuning for New Languages
A comprehensive toolkit for fine-tuning Coqui's XTTS model to support new languages, with a focus on low-resource languages like Maltese. This implementation extends XTTS's tokenizer using BPE (Byte-Pair Encoding) and provides various training strategies to minimize catastrophic forgetting.

&nbsp;  
## Features
- **Tokenizer Extension**: Automated BPE-based tokenizer extension for new languages
- **Multi-Dataset Support**: Train on multiple datasets simultaneously
- **Forgetting Mitigation**: LoRA and layer freezing options to preserve existing language capabilities
- **Flexible Training**: Support for both GPT and DVAE fine-tuning
- **Maltese Language Support**: Specialized preprocessing for Maltese text
- **Python 3.11+ Compatibility**: Includes compatibility layer for newer Python versions

&nbsp;  
## Training Parameters
### Essential Parameters
- `--num_epochs`: Training cycles (start with 10-50 for testing, 100+ for production)
- `--batch_size`: Samples per batch (reduce if OOM, typically 2-8)
- `--grad_acumm`: Gradient accumulation steps (increase if batch_size is small)
- `--lr`: Learning rate (5e-6 is safe, 1e-5 for faster convergence)

### Forgetting Mitigation Strategies
- `NONE`: Standard fine-tuning (highest risk of forgetting)
- `LORA`: Low-Rank Adaptation (recommended, preserves original capabilities)
- `FREEZE`: Freezes most layers (most conservative, slower adaptation)

&nbsp;  
## File Structure
- `compatibility.py`: Python 3.11+ compatibility layer
- `download.py`: Model downloader
- `tokenizer_extension.py`: BPE tokenizer extension
- `trainingGPT.py`: GPT model training
- `trainingDVAE.py`: DVAE model training
- `inference.py`: Text-to-speech synthesis
- `prepare_maltese_dataset.py`: Maltese dataset preparation
- `new_language_training_cli.py`: All-in-one training script
- `utils.py`: Utility functions
- `parsers.py`: Command-line argument parsers


&nbsp;  
## 1. Installation
First, clone the repository and install the necessary dependencies:  
```bash
git clone https://github.com/Wubpooz/Malta-TTS.git
cd Malta-TTS/FineTuning/NewLanguage

python -m venv venv
# Switch to the virtual environment
venv/Scripts/activate # For Windows
# or
source venv/bin/activate # For Linux

pip install --upgrade pip
pip install -r requirements.txt
pip install tf-keras tensorflow-decision-forests tensorflow-text --upgrade

python -c "import stanza; stanza.download('mt')"
```

&nbsp;  
&nbsp;  
## 2. Data Preparation
Organize your dataset as follows:
```
root/
├── datasets-1/
│   ├── wavs/
│   │   ├── xxx.wav
│   │   ├── yyy.wav
│   │   ├── zzz.wav
│   │   └── ...
│   ├── metadata_train.csv
│   ├── metadata_eval.csv
├── datasets-2/
│   ├── wavs/
│   │   ├── xxx.wav
│   │   ├── yyy.wav
│   │   ├── zzz.wav
│   │   └── ...
│   ├── metadata_train.csv
│   ├── metadata_eval.csv
...
```

Format your `metadata_train.csv` and `metadata_eval.csv` files as follows:  
```
audio_file|text|normalized_text|speaker_name
name1|transcription1|normalized_transcription1|speaker1
name2|transcription2|normalized_transcription2|speaker1
...
```

&nbsp;  
You may also use the provided `prepare_maltese_dataset.py` script to automate this process:
```bash
python prepare_maltese_dataset.py \
    --input_dir /path/to/MASRI_HEADSET_v2 \
    --output_dir /path/to/output \
    --test_size 0.2
```


&nbsp;  
&nbsp;  
## Direct Finetuning with One Command
You can directly finetune the model using the `new_language_training_cli.py` script. This script will handle downloading the base model, extending the tokenizer, and training the GPT model in one go. Here’s how to use it:

```bash
env TOKENIZERS_PARALLELISM=false
env OMP_NUM_THREADS=1
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python new_language_training_cli.py \
  --is_download True \
  --output_path checkpoints/ \
  --metadatas datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,mt datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,mt \
  --language mt \
  --num_epochs 100 \
  --batch_size 3 \
  --grad_acumm 84 \
  --min_frequency 2 \
  --max_new_tokens 8000 \
  --max_audio_length 255995 \
  --max_text_length 200 \
  --weight_decay 1e-2 \
  --lr 5e-6 \
  --save_step 10000 \
  --version main \
  --forgetting_mitigation LORA \
  --optimizations True \
  --tf32 True \
  --multi_gpu True
```


&nbsp;  
&nbsp;  
## Manual Finetuning Steps
### (Optional) Downloading pretrained model manually
Execute the following command to download the pretrained model:  
```bash
python download.py --output_path checkpoints/ --version main
```
Optionally, specify a custom model:  
```bash
python download.py --output_path checkpoints/ --version main --custom_model custom_model_name.pth
```
It needs the version of the model to download config and vocab files.  


 
&nbsp;  
### (Optional) Extend the tokenizer manually
Execute the following command to extend the tokenizer with a new language (requires the `vocab.json` and `config.json` files from the pretrained model):  
```bash
python tokenizer_extension.py \
  --output_path checkpoints/ \
  --xtts_checkpoint checkpoints/model.pth \
  --tokenizer_file checkpoints/vocab.json \
  --config_path checkpoints/config.json \
  --metadata_path datasets/metadata_train.csv \
  --language mt \
  --extended_vocab_size 2000
```
You can also control frequency thresholds and cap the number of new tokens:
```bash
python tokenizer_extension.py ... --min_frequency 2 --max_new_tokens 8000
```


&nbsp;  
### GPT Finetuning
```bash
Run training directly (without re-downloading or tokenizer extension):
```bash
python trainingGPT.py \
  --output_path checkpoints/ \
  --metadatas datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,mt datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,mt \
  --language mt \
  --mel_norm_file checkpoints/mel_stats.pth \
  --dvae_checkpoint checkpoints/dvae.pth \
  --xtts_checkpoint checkpoints/model.pth \
  --tokenizer_file checkpoints/vocab.json \
  --vocab_size 6855 \
  --num_epochs 100 \
  --batch_size 3 \
  --grad_acumm 84 \
  --lr 5e-6 \
  --weight_decay 1e-2 \
  --save_step 10000 \
  --max_audio_length 255995 \
  --max_text_length 200 \
  --forgetting_mitigation LORA \
  --optimizations True \
  --tf32 True \
  --multi_gpu True
```


&nbsp;  
### (Optional) DVAE Finetuning
You can also train the Discrete VAE (DVAE) model with the following command:  
```bash
python trainingDVAE.py \
  --dvae_pretrained checkpoints/dvae.pth \
  --mel_norm_file checkpoints/mel_stats.pth \
  --language mt \
  --metadatas datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,mt datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,mt \
  --num_epochs 5 \
  --batch_size 512 \
  --lr 5e-6
```



&nbsp;  
&nbsp;  
## 4. Inference
The inference can be run with the following command:  
```bash
python inference.py \
  --xtts_checkpoint checkpoints/model.pth \
  --xtts_config checkpoints/config.json \
  --xtts_vocab checkpoints/vocab.json \
  --tts_text "Hello, this is a test." \
  --speaker_audio_file path/to/speaker/audio.wav \
  --lang mt \
  --output_file output.wav \
  --temperature 0.7 \
  --repetition_penalty 1.05 \
  --length_penalty 10.0 \
  --top_k 50 \
  --top_p 0.8 \
  --LORA_trained
```


--- 

&nbsp;  
&nbsp;  
## Notes
Finetuning the HiFiGAN decoder was attempted by `anhnh2002` for Vietnamese but resulted in worse performance. DVAE and GPT finetuning are sufficient for optimal results. They also found that ff you have enough short texts in your datasets (about 20 hours), you do not need to finetune DVAE.  

&nbsp;  
## Troubleshooting
### Out of Memory
- Reduce `batch_size` and increase `grad_acumm`
- Enable mixed precision: add `--optimizations`
- Use LoRA: `--forgetting_mitigation LORA`

### Poor Quality Output
- Increase training epochs
- Check dataset quality (clean audio, accurate transcriptions)
- Adjust inference parameters (temperature, repetition_penalty)

### Model Forgetting Other Languages
- Use LoRA or FREEZE strategies
- Include some English/original language data in training
- Lower learning rate


&nbsp;  
&nbsp;  
## Citation
If you use this code, please cite:
```bibtex
@software{xtts_maltese_2024,
  title={XTTS Fine-tuning for Maltese},
  author={Wubpooz},
  year={2024},
  url={https://github.com/Wubpooz/Malta-TTS}
}
```

&nbsp;  
&nbsp;  
## Acknowledgments
- [Coqui TTS](https://github.com/coqui-ai/TTS) for the XTTS model
- [MASRI](https://mlrs.research.um.edu.mt/) for Maltese language resources
- Based on work by [anhnh2002](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages)