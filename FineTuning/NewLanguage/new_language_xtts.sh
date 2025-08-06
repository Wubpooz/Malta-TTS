# -*- coding: utf-8 -*-
# new_language_xtts.py
# This script is used to fine-tune the XTTS model for a new language and run inference on it.
# It includes functions for training the model and performing inference, as well as a command-line interface to
# control the process.
# Based on https://github.com/daswer123/xtts-webui/blob/main/scripts/utils/gpt_train.py & https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages/blob/main/train_gpt_xtts.py


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/coqui-ai/TTS.git@dev
pip install -r requirements.txt


python -c "import nltk; nltk.download('punkt')"
python -m spacy download en_core_web_sm # XTTS uses spacy for some languages. Even if not "mt"


# CUDA_VISIBLE_DEVICES=0 python dataset_preparation.py \
# --input_path datasets/ \
# --output_path datasets-1/ \
# --language mt \
# --metadata_path datasets/metadata_train.csv \


# CUDA_VISIBLE_DEVICES=0
python new_language_training_cli.py \
--output_path checkpoints/ \
--metadatas datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,mt datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,mt \
--num_epochs 100 \
--batch_size 8 \
--grad_acumm 84 \
--max_text_length 200 \
--max_audio_length 255995 \
--weight_decay 1e-2 \
--lr 5e-6 \
--save_step 10000 \
--custom_model "" \
--version main \
--metadata_path datasets/metadata_train.csv \
--language mt \
--extended_vocab_size 2000

