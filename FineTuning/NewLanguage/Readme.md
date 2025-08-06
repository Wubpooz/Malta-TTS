# XTTS Finetuning Guide for New Languages
## 1. Installation
First, clone the repository and install the necessary dependencies:  
```
git clone https://github.com/nguyenhoanganh2002/XTTSv2-Finetuning-for-New-Languages.git
cd XTTSv2-Finetuning-for-New-Languages
pip install -r requirements.txt
```

## 2. Data Preparation
Ensure your data is organized as follows:
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
│   
├── recipes/
├── scripts/
├── TTS/
└── README.md
```

Format your `metadata_train.csv` and `metadata_eval.csv` files as follows:  
```
audio_file|text|speaker_name
wavs/xxx.wav|How do you do?|@X
wavs/yyy.wav|Nice to meet you.|@Y
wavs/zzz.wav|Good to see you.|@Z
```

## 3. Pretrained Model Download
Execute the following command to download the pretrained model:  
```bash
python download_checkpoint.py --output_path checkpoints/
```

## 4. Vocabulary Extension and Configuration Adjustment
Extend the vocabulary and adjust the configuration with:  
```bash
python extend_vocab_config.py --output_path=checkpoints/ --metadata_path datasets/metadata_train.csv --language mt --extended_vocab_size 2000
```

## 5. DVAE Finetuning (Optional)
To finetune the DVAE, run:  
```bash
CUDA_VISIBLE_DEVICES=0 python train_dvae_xtts.py \
--output_path=checkpoints/ \
--train_csv_path=datasets/metadata_train.csv \
--eval_csv_path=datasets/metadata_eval.csv \
--language="vi" \
--num_epochs=5 \
--batch_size=512 \
--lr=5e-6
```

## 6. GPT Finetuning
```bash
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
--output_path checkpoints/ \
--metadatas datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,vi datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,vi \
--num_epochs 5 \
--batch_size 8 \
--grad_acumm 4 \
--max_text_length 400 \
--max_audio_length 330750 \
--weight_decay 1e-2 \
--lr 5e-6 \
--save_step 50000
```



Note: Finetuning the HiFiGAN decoder was attempted but resulted in worse performance. DVAE and GPT finetuning are sufficient for optimal results.  
Update: If you have enough short texts in your datasets (about 20 hours), you do not need to finetune DVAE.  
