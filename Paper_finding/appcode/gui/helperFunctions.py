import pandas as pd
import re

def tuple_list_to_df(tuple_list):
  df_data = []
  for item in tuple_list:
    if len(item) == 3: # Already has enabled state
      df_data.append({"Pattern": item[0], "Weight": item[1], "Enabled": item[2]})
    else: # Old format, assume enabled
      df_data.append({"Pattern": item[0], "Weight": item[1], "Enabled": True})
  return pd.DataFrame(df_data, columns=["Enabled", "Pattern", "Weight"]) # 'Enabled' first for visibility

def df_to_tuple_list(df):
  return [(row["Pattern"], int(row["Weight"]), row["Enabled"]) for _, row in df.iterrows() if row["Pattern"].strip() and row["Enabled"]]

# --- Optional regex pattern auto-generation for keywords ---
def auto_regexify(pattern_str):
  """
  Converts simple phrases to regex: 'text to speech' -> '\\btext[- ]to[- ]speech\\b'
  Leaves raw regex (with backslashes or regex syntax) untouched.
  """
  if re.search(r"[\\\[\]\(\)\|\.\+\*\?]", pattern_str):  # looks like raw regex
    return pattern_str
  tokens = re.split(r"\s+", pattern_str.strip())
  return r"\b" + r"[- ]".join(tokens) + r"\b"

def convert_column_to_regex(df):
  df["Pattern"] = df["Pattern"].apply(auto_regexify)
  return df

