# XTTS Finetuning for New Languages (e.g., Maltese)
**Path passed to this script can be either relative or absolute.**

## 1. Installation
First, clone the repository and install the necessary dependencies:  
```
git clone https://github.com/Wubpooz/Malta-TTS.git
cd Malta-TTS/FineTuning/NewLanguage

python -m venv venv
# Switch to the virtual environment
venv/Scripts/activate # For Windows
# or
source venv/bin/activate # For Linux

pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# pip install git+https://github.com/coqui-ai/TTS.git@dev
python -c "import nltk; nltk.download('punkt')"
python -m spacy download en_core_web_sm # XTTS uses spacy for some languages. Even if not "mt"
```

&nbsp;  
&nbsp;  
## 2. Data Preparation
Either use `prepare_maltese_dataset.py` to prepare your dataset or manually organize your data as follows:
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
audio_file|text|speaker_name
wavs/1.wav|transcription1|speaker1
wavs/2.wav|transcription2|speaker1
...
```


&nbsp;  
&nbsp;  
## (Optional) Downloading pretrained model manually
Execute the following command to download the pretrained model:  
```bash
python download.py --output_path=checkpoints/ --version=main
```
Optionnally, you can specify a custom model:  
```bash
python download.py --output_path=checkpoints/ --version=main --custom_model=custom_model_name
```
It needs the version still to download config and vocab files.  


&nbsp;  
&nbsp;  
## (Optional) Extend the tokenizer manually
Execute the following command to extend the tokenizer with a new language (Requires the `vocab.json` file from the pretrained model):  
```bash
python tokenizer_extension.py --output_path=checkpoints/ --metadata_path=datasets/metadata_train.csv --language mt --extended_vocab_size 2000
```
Extended vocabulary size is the maximum number of tokens to be added to the tokenizer, in version 2.0 XTTS seems to have around 7,000 tokens.  


&nbsp;  
&nbsp;  
## 3. GPT Finetuning
To run the finetuning with download and tokenizer extension, execute the following command:  
```bash
python new_language_training_cli.py --is_download True --is_tokenizer_extension True --output_path=checkpoints/ --metadatas=datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,mt datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,mt --num_epochs 100 --batch_size 3 --grad_acumm 84 --max_audio_length 255995 --max_text_length 200 --weight_decay 1e-2 --lr 5e-6 --save_step 10000 --custom_model=custom_model_name --version=main --multi_gpu --metadata_path=datasets/metadata_train.csv --language mt --extended_vocab_size 2000
```

&nbsp;  
The training can also be run without the download and tokenizer extension:  
```bash
python trainingGPT.py --output_path=checkpoints/ --metadatas=datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,mt datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,mt --num_epochs 100 --batch_size 3 --grad_acumm 84 --max_audio_length 255995 --max_text_length 200 --weight_decay 1e-2 --lr 5e-6 --save_step 10000 --custom_model=custom_model_name --version=main --multi_gpu
```


&nbsp;  
&nbsp;  
## (Optional) DVAE Finetuning



&nbsp;  
&nbsp;  
## 4. Inference
The inference can be run with the following command:  
```bash
python inference.py --xtts_checkpoint=checkpoints/xtts.pth --xtts_config=checkpoints/config.json --xtts_vocab=checkpoints/vocab.json --tts_text="Hello, this is a test." --speaker_audio_file="path/to/speaker/audio.wav" --lang="mt" --output_file="output.wav"
```


--- 

&nbsp;  
&nbsp;  
## Notes
Finetuning the HiFiGAN decoder was attempted by `anhnh2002` for Vietnamese but resulted in worse performance. DVAE and GPT finetuning are sufficient for optimal results. They also found that ff you have enough short texts in your datasets (about 20 hours), you do not need to finetune DVAE.  




## trainingGPT calls download => make that optional but require file path then