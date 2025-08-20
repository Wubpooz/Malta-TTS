from . import lang_MT

def num2text(number, ordinal=False, lang='mt', to='cardinal', **kwargs):
    if lang !="mt":
      raise NotImplementedError(f"Language '{lang}' is not supported for num2text conversion, it's a maltese only implementation of num2word. Use num2word instead.")
    converter = lang_MT.Num2Word_MT()
    number = str(number)
    number = number.replace(",", "")
    if isinstance(number, str):
      number = converter.str_to_number(number)

    if ordinal:
        to = 'ordinal'

    if to not in ['cardinal', 'ordinal', 'ordinal_num', 'year', 'currency']:
        raise NotImplementedError(f"Conversion type '{to}' is not supported. Supported types are: 'cardinal', 'ordinal', 'ordinal_num', 'year', 'currency'.")

    return getattr(converter, 'to_{}'.format(to))(number, **kwargs)

