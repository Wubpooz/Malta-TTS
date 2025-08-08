# ============================ Requirements ============================
# - Python 3.8 or higher
# - PyTorch with CUDA support
# - Required Python packages listed in requirements.txt
python -m venv venv
#find if on windows or linux:
if [ -d "venv/Scripts" ]; then
  echo "Windows detected, using venv/Scripts/activate"
  venv/Scripts/activate
else
  echo "Linux detected, using venv/bin/activate"
  source venv/bin/activate
fi

pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# pip install git+https://github.com/coqui-ai/TTS.git@dev
python -c "import nltk; nltk.download('punkt')"
python -m spacy download en_core_web_sm # XTTS uses spacy for some languages. Even if not "mt"




# ============================ Dataset Preparation ============================
# CUDA_VISIBLE_DEVICES=0 python dataset_preparation.py \
# --input_path datasets/ \
# --output_path datasets-1/ \
# --language mt \
# --metadata_path datasets/metadata_train.csv \


# ============================ Model Training ============================
# CUDA_VISIBLE_DEVICES=0
python new_language_training_cli.py \
--is_download \
--is_tokenizer_extension \
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

