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
## add more error handling
## freeze requirements
## fix paths and params
## cleanup






## training loss graph
## autodetect formatter based on metadata number of cols and names

## Logger?
import logging
logger = logging.getLogger(__name__)



## mixed precision










Great question — and Maltese is a perfect example of what “portable TTS” really needs: robustness with tiny models and smart data tricks. Here’s a practical, end-to-end blueprint you can follow, with options from “works on a Raspberry Pi/phone” to “bigger but still deployable”.

1) Choose a portable-friendly architecture
	•	Piper (Glow-TTS/HiFi-GAN–style, C++ runtime): tiny, fast, proven on ARM; easy to train your own voices.
	•	VITS / FastPitch + HiFi-GAN (light): great quality; prune + quantize for edge.
	•	Distilled WaveRNN vocoder (if you need ultra-low CPU without GPU).
	•	On-device runtimes: ONNX Runtime (mobile/ARM), Core ML (iOS), TFLite (Android), or a pure C/C++ runtime (Piper).

2) Data Augmentations
	•	Speed perturb (0.9×/1.0×/1.1×), room IRs, light noise, pitch shift ≤ ±100 cents (sparingly), SpecAugment on spectrograms for the acoustic model.
	•	Add code-switch examples (Maltese + English/Italian) — realistic for Malta and improves robustness.

3) Front-end: text normalization + G2P that actually works for Maltese
Maltese orthography is close to phonemic, but you still need rules:
	•	Build a rule-based G2P first (grapheme→phoneme mappings + stress heuristics).
	•	Maintain a lexicon for exceptions, names, loanwords, and abbreviations.
	•	Use a unified phoneme set (IPA or X-SAMPA). Make sure the vocoder/acoustic model uses the same symbol IDs across languages.
	•	Fallback path: if a token is OOV and ambiguous, back off to graphemes (models learn this surprisingly well) or byte-level pieces.

4) Training recipes (concrete)
A. Small single-speaker Maltese voice (fastest path)
	•	Model: FastPitch (acoustic) + HiFi-GAN (vocoder, V1 light).
	•	Hours: 2–10 h clean, single speaker.
	•	Steps
	1.	Train HiFi-GAN on multilingual data first (transferable), then fine-tune on your Maltese speaker (50k–200k steps).
	2.	Train FastPitch on phonemes; use duration/pitch predictors; batch size small (e.g., 16–32 on a single GPU).
	3.	Early stop by MOS proxy: ASR-CER on TTS–>ASR, and an external prosody score.

B. Multispeaker Maltese (+ neighbors) for robustness
	•	Model: VITS (multispeaker) with speaker embeddings; or Piper multispeaker recipe.
	•	Data: combine Maltese with Italian/Arabic/English; ensure at least ~30–60 min per speaker for stability.
	•	Loss tricks: feature matching (for HiFi-GAN), duration loss with stochastic duration predictor (VITS), mild speaker-mixup.

C. Distill/convert for portability
	•	Export acoustic to ONNX (opset 17+), export vocoder to ONNX or keep a C++ HiFi-GAN.
	•	Quantize:
	•	Dynamic INT8 for linears/conv1d (ONNX Runtime).
	•	For mobile GPUs/NNAPI/Core ML: try 16-bit float or 8-bit weight-only.
	•	Aim: < 100 MB total (acoustic ≤ 30–60 MB, vocoder ≤ 40 MB).

Implementation tips
	•	Streaming synthesis: chunk text (clauses), synth acoustic chunks, stream vocoder frames as they’re ready.
	•	Use smaller hop size (256) at 22.05 kHz to balance latency/naturalness.
	•	Pre-warm models on app start (first inference JIT costs).
	•	Cache phonemized text and punctuation normalization.

5) Maltese specifics that help quality
	•	Handle Maltese letters (ċ, ġ, għ, ħ, ż) carefully in normalization.
	•	Prosody: Maltese stress tends toward the penultimate syllable — add a heuristic to your G2P (and learn residual stress via FastPitch).
	•	Loanwords: keep dual lexicon entries (native vs. borrowed pronunciation); pick by context (neighbor tokens, language ID tags).
	•	Code-switch tags: mark spans like <lang=en>, <lang=it> to signal different phonotactics without swapping models.

6) Evaluation & QA (cheap but effective)
	•	Objective: ASR-CER of TTS audio using a strong multilingual ASR (Whisper small/distilled) — lower is better.
	•	Prosody: F0 variance vs. references; pause placement around punctuation.
	•	Subjective: 5–10 native Maltese listeners; MOS + ABX against ground truth; include code-switch sentences and numerics/dates.
	•	Robustness set: typos, all-caps, abbreviations, URLs, telephone numbers, currency, times.

7) A minimal training & export sketch (framework-agnostic)


##### Steps
 1) Train vocoder (HiFi-GAN light) on multilingual + fine-tune on Maltese
python train_hifigan.py --config hifigan_light.yaml --data data_multilingual
python finetune_hifigan.py --checkpoint ckpt_hifigan.pth --data data_mt_speaker

 2) Train FastPitch on phonemes
python train_fastpitch.py --config fastpitch_mt.yaml --train-metadata metadata_mt_phoneme.csv

 3) Export to ONNX
python export_fastpitch_onnx.py --checkpoint fastpitch_mt.pth --out fastpitch_mt.onnx
python export_hifigan_onnx.py --checkpoint hifigan_mt.pth --out hifigan_mt.onnx

 4) Quantize (dynamic int8)
python onnx_quantize.py --in fastpitch_mt.onnx --out fastpitch_mt_int8.onnx --mode dynamic
python onnx_quantize.py --in hifigan_mt.onnx --out hifigan_mt_int8.onnx --mode dynamic

9) Packaging for apps
	•	Android: bundle ONNX models, run with ONNX Runtime (NNAPI/GPU EP if available). Kotlin wrapper; stream PCM to AudioTrack.
	•	iOS: convert to Core ML (coremltools), or use ONNX Runtime Mobile; stream to AVAudioEngine.
	•	Desktop/IoT: Piper or your own C++ runner; one static binary + two model files.

10) If you already have a big multilingual TTS (e.g., XTTS)
	•	Use it to bootstrap: generate high-quality Maltese pseudo-data (careful: avoid style collapse), then distill into a small FastPitch/VITS student on phonemes.
	•	Keep the big model server-side for rare names or tricky code-switches; default to on-device small model; fall back to server only when the small model flags low confidence.


https://www.reddit.com/r/mlscaling/comments/1gxakk3/did_a_quick_comparison_of_various_tts_models/