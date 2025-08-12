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
You can also train the Discrete VAE (DVAE) model with the following command:  
```bash
python trainingDVAE.py --dvae_pretrained checkpoints/dvae.pth --mel_norm_file checkpoints/mel_norm.json --language mt --metadatas=datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,mt datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,mt --num_epochs 5 --batch_size 512 --lr 5e-6
```



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



# TODOs
## trainingGPT calls download => make that optional but require file path then
## maybe audio file khz inconsistencies
## use global paths in the colab notebook at the top
## do the TODOS in files
## param for num_workers ?
## training loss graph
## Use deepspeed
If you want to be able to load_checkpoint with use_deepspeed=True and enjoy the speedup, you need to install deepspeed first.
`pip install deepspeed==0.10.3`

## autodetect formatter based on metadata number of cols and names





## mixed precision:
1. Add a mixed_precision flag to your GPTTrainerConfig
python
Copy
Edit
@dataclass
class GPTTrainerConfig(XttsConfig):
    lr: float = 5e-06
    training_seed: int = 1
    optimizer_wd_only_on_weights: bool = False
    weighted_loss_attrs: dict = field(default_factory=lambda: {})
    weighted_loss_multipliers: dict = field(default_factory=lambda: {})
    test_sentences: List[dict] = field(default_factory=lambda: [])
    mixed_precision: bool = False  # <<< NEW
2. Modify train_step to support AMP
Replace:

python
Copy
Edit
loss_text, loss_mel, _ = self.forward(
    text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens
)
with:

python
Copy
Edit
if self.config.mixed_precision and self.device.type == "cuda":
    with torch.cuda.amp.autocast(dtype=torch.float16):
        loss_text, loss_mel, _ = self.forward(
            text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens
        )
else:
    loss_text, loss_mel, _ = self.forward(
        text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens
    )
3. Add gradient scaling in the training loop
If your training loop is inside the Trainer class (from trainer package), you’ll need something like:

python
Copy
Edit
scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

for batch in train_loader:
    optimizer.zero_grad()

    with torch.cuda.amp.autocast(enabled=config.mixed_precision):
        _, loss_dict = model.train_step(batch, criterion)
        loss = loss_dict["loss"]

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
That way, when mixed_precision=True in your config, the model runs FP16 forward passes and scaled gradients, but still keeps FP32 master weights.

4. Activate it in your fine-tune script
When you build the GPTTrainerConfig in your train_gpt():

python
Copy
Edit
config = GPTTrainerConfig(
    ...
    mixed_precision=True,  # enable AMP
)
This will:

Cut VRAM usage ~40–50% (helpful for Colab T4).

Possibly increase speed (depends on compute/memory balance).

Automatically fall back to FP32 for numerically sensitive ops.