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




def add_language_to_tokenizer(tokenizer, lang_code="mt"):
  """
    Adds your language to the tokenizer.py file using Monkey Patching.
    # pip install spacy stanza spacy-stanza
    # python -c "import stanza; stanza.download('mt')"
  """
  import re
  import spacy_stanza
  from masri.transcribe.num2text import num2text
  from masri.tokenise.tokenise import MTWordTokenizer, MTRegex
  
  mt_word_tokenizer = MTWordTokenizer()


  _original_get_spacy_lang = tokenizer.get_spacy_lang
  def get_spacy_lang(lang):
    if lang == lang_code:
      return spacy_stanza.load_pipeline(lang_code, processor="tokenize")
    else:
      return _original_get_spacy_lang(lang)

  tokenizer.get_spacy_lang = get_spacy_lang 


  tokenizer._abbreviations[lang_code] = [
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


  tokenizer._symbol_multilingual[lang_code] = [
    (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
    for x in [
      ("&", " u "),
      ("@", " fuq "),
      ("%", " percent "),
      ("#", " hash "),
      ("$", " dollar "),
      ("£", " pound "),
      ("°", " degree "),
    ]
  ]

  tokenizer._ordinal_re[lang_code] = re.compile(MTRegex.DEF_NUMERAL)


  if not hasattr(tokenizer, "char_limits"):
    tokenizer.char_limits = {}
  if lang_code not in tokenizer.char_limits:
    tokenizer.char_limits[lang_code] = tokenizer.char_limits.get("en", 400)
    print(f"Added char_limits for '{lang_code}' language.")



  original_expand_numbers_multilingual = tokenizer.expand_numbers_multilingual
  def custom_expand_numbers_multilingual(text, lang):
    if lang == lang_code:
      text = num2text(text)
    return original_expand_numbers_multilingual(text, lang)
  
  tokenizer.expand_numbers_multilingual = custom_expand_numbers_multilingual


  _original_expand_currency = tokenizer._expand_currency
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
    
  tokenizer._expand_currency = _custom_expand_currency

  

  _original_preprocess_text = tokenizer.VoiceBpeTokenizer.preprocess_text
  def custom_preprocess_text(self, txt, lang):
    if lang == lang_code:
      txt = mt_word_tokenizer.tokenize_fix_quotes(txt)
      if isinstance(txt, list):
        txt = " ".join(txt)
      return tokenizer.multilingual_cleaner(txt, lang)
      # TODO transliterate ?
    return _original_preprocess_text(self, txt, lang)

  tokenizer.VoiceBpeTokenizer.preprocess_text = custom_preprocess_text

  print(f"{lang_code} added to tokenizer.py!")