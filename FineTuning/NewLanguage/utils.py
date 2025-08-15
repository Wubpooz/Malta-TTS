
def add_language_to_VoiceBpeTokenizer(lang_code: str):
  """Add a new language to the tokenizer.
  Args:
      lang (str): Language code for the new language to be added.
  """
  import re
  import TTS.tts.layers.xtts.tokenizer as tokenizer
  print(f"Adding new language: {lang_code}")

  _original_preprocess_text = tokenizer.VoiceBpeTokenizer.preprocess_text

  def custom_preprocess_text(self, txt, lang):
    if lang == lang_code:
      txt = txt.lower()
      txt = re.sub(re.compile(r"\s+"), " ", txt)
      # TODO transliterate ?
      return txt.strip()
    return _original_preprocess_text(self, txt, lang)

  # Monkey-patch
  tokenizer.VoiceBpeTokenizer.preprocess_text = custom_preprocess_text
  print(f"New language added: {lang_code}")
