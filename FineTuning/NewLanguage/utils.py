import compatibility

def preprocess_maltese_text(text):
  """Enhanced Maltese text preprocessing using MTWordTokenizer"""
  from masri.tokenise.tokenise import MTWordTokenizer, MTRegex

  tokenizer = MTWordTokenizer()

  text = tokenizer.tokenize_fix_quotes(text)
  
  # Join tokens back but with proper spacing for clitics
  if isinstance(text, list):
    processed_tokens = []
    i = 0
    while i < len(text):
      token = text[i]      
      # Check if current token is a clitic that should attach to next token
      if MTRegex.is_prefix(token) and i + 1 < len(text):
        # Join with next token
        processed_tokens.append(token + text[i + 1])
        i += 2
      else:
        processed_tokens.append(token)
        i += 1

    return " ".join(processed_tokens)
  else:
    return " ".join(tokenizer.tokenize(text))




def add_language_to_tokenizer(VoiceBPETokenizer, lang_code="mt"):
  """
    Adds your language to the tokenizer.py file using Monkey Patching.
    # pip install spacy stanza spacy-stanza
    # python -c "import stanza; stanza.download('mt')"
  """
  import TTS.tts.layers.xtts.tokenizer as tokenizerFile

  import re
  import spacy_stanza
  from masri.transcribe.num2text import num2text
  from masri.tokenise.tokenise import MTWordTokenizer, MTRegex
  
  mt_word_tokenizer = MTWordTokenizer()


  _original_get_spacy_lang = tokenizerFile.get_spacy_lang
  def get_spacy_lang(lang):
    if lang == lang_code:
      return spacy_stanza.load_pipeline(lang_code, processor="tokenize")
    else:
      return _original_get_spacy_lang(lang)

  tokenizerFile.get_spacy_lang = get_spacy_lang 


  tokenizerFile._abbreviations[lang_code] = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
      ("sra", "sinjura"),
      ("sr", "sinjur"),
      ("dr", "doktor"),
      ("dra", "doktri"),
      ("st", "santu"),
      ("co", "kumpanija"),
      ("jr", "junior"),
      ("ltd", "limitata"),
    ]
  ]


  tokenizerFile._symbols_multilingual[lang_code] = [
    (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
    for x in [
      ("&", " u "),
      ("@", " fuq "),
      ("%", " ful-mija "),
      ("#", " hash "),
      ("$", " dollaru "),
      ("£", " lira "),
      ("°", " grad "),
    ]
  ]

  tokenizerFile._ordinal_re[lang_code] = re.compile(MTRegex.DEF_NUMERAL)


  if not hasattr(VoiceBPETokenizer, "char_limits"):
    VoiceBPETokenizer.char_limits = {}
  if lang_code not in VoiceBPETokenizer.char_limits:
    VoiceBPETokenizer.char_limits[lang_code] = VoiceBPETokenizer.char_limits.get("en", 400)
    print(f"Added char_limits for '{lang_code}' language.")



  _original_expand_decimal_point = tokenizerFile._expand_decimal_point
  def custom_expand_decimal_point(m, lang="en"):
    if lang == lang_code:
      return num2text(float(m.group(1)), lang=lang)
    else:
      return _original_expand_decimal_point(m, lang)

  tokenizerFile._expand_decimal_point = custom_expand_decimal_point


  _original_expand_currency = tokenizerFile._expand_currency
  def _custom_expand_currency(m, lang="en", currency="USD"):
    if lang == lang_code:
      amount = float((re.sub(r"[^\d.]", "", m.group(0).replace(",", "."))))
      full_amount = num2text(amount, to="currency", currency=currency, lang=lang)
      if amount.is_integer():
        last_and = full_amount.rfind(", ")
        if last_and != -1:
          full_amount = full_amount[:last_and]

      return full_amount

    else:
      return _original_expand_currency(m, lang, currency)

  tokenizerFile._expand_currency = _custom_expand_currency


  _original_expand_ordinal = tokenizerFile._expand_ordinal
  def custom_expand_ordinal(m, lang="en"):
    if lang == lang_code:
      return num2text(int(m.group(1)), ordinal=True, lang=lang)
    else:
      return _original_expand_ordinal(m, lang)

  tokenizerFile._expand_ordinal = custom_expand_ordinal


  _original_expand_number = tokenizerFile._expand_number
  def custom_expand_number(m, lang="en"):
    if lang == lang_code:
      return num2text(int(m.group(0)), lang=lang)
    else:
      return _original_expand_number(m, lang)

  tokenizerFile._expand_number = custom_expand_number


  # original_expand_numbers_multilingual = tokenizerFile.expand_numbers_multilingual
  # def custom_expand_numbers_multilingual(text, lang):
  #   if lang == lang_code:
  #     try:
  #       text = num2text(text)
  #     except Exception as e:
  #       print(f"Error in num2text: {e}")
  #       print(f"Text to expand: {text}")
  #       exit(1)
  #   return original_expand_numbers_multilingual(text, lang)
  
  # tokenizerFile.expand_numbers_multilingual = custom_expand_numbers_multilingual





  _original_preprocess_text = VoiceBPETokenizer.preprocess_text
  def custom_preprocess_text(self, txt, lang):
    if lang == lang_code:
      txt = mt_word_tokenizer.tokenize_fix_quotes(txt)
      if isinstance(txt, list):
        txt = " ".join(txt)
      return tokenizerFile.multilingual_cleaners(txt, lang)
      # TODO transliterate ?
    return _original_preprocess_text(self, txt, lang)

  tokenizerFile.VoiceBpeTokenizer.preprocess_text = custom_preprocess_text

  print(f"{lang_code} added to tokenizer.py!")