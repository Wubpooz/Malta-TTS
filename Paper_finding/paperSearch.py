import re
import os

def find_and_score_titles(base_path, weighted_patterns, output_file=None, EXCLUDE_SOURCES = ("theme_matches", "output_", "log_")):
  """
  Finds and scores paper titles based on weighted patterns from text files in a specified directory.

  Args:
      base_path (str): The path to the directory containing text files with paper titles.
      weighted_patterns (list of tuples): A list of tuples where each tuple contains a regex pattern 
                                          and its associated weight for scoring.
      output_file (str, optional): The name of the file to save the scored paper titles. Defaults to 'theme_matches.txt'.
      EXCLUDE_SOURCES (tuple, optional): A tuple of file name prefixes to exclude from processing. Defaults to 
                                          ("theme_matches", "output_", "log_").

  Returns:
      None. Outputs the results by printing and saving scored titles to a text file.

  This function reads text files from the specified directory, applies regex patterns to the titles to compute scores,
  filters out titles from excluded sources, removes duplicates while retaining the highest score, sorts the titles 
  by score in descending order, and saves the results to a file. It also prints the top matching titles and the
  number of matches found.
  """

  txt_files = [f for f in os.listdir(base_path) if f.endswith(".txt") and not f.startswith(EXCLUDE_SOURCES)]
  print(f"Found {len(txt_files)} text files in {base_path}.")

  papers_by_file = {}
  for txt_file in txt_files:
    print(f"Loading {txt_file}...")
    file_path = os.path.join(base_path, txt_file)
    with open(file_path, "r", encoding="utf-8") as f:
      lines = [line.strip() for line in f if line.strip()]
      key = os.path.splitext(txt_file)[0]
      papers_by_file[key] = lines
    print(f"Loaded {len(papers_by_file[key])} lines from {txt_file}.")

  print(f"Loaded {len(papers_by_file)} files.")
  if not papers_by_file:
    print("No papers found. Exiting.")
    return

  print("Compiling patterns...")
  compiled_weighted_patterns = [(re.compile(pat, re.IGNORECASE), weight) for pat, weight in weighted_patterns]

  print("Scoring titles...")
  theme_matches = []
  for source, titles in papers_by_file.items():
    for title in titles:
      #TODO try to normalize the title, but they can vary from source to source, include keywords for icassp for example
      score = sum(weight for pattern, weight in compiled_weighted_patterns if pattern.search(title))
      if score > 0:
        theme_matches.append((score, source, title))
  print(f"Found {len(theme_matches)} matches before filtering excluded sources and removing duplicates.")

  theme_matches = list(filter(lambda x: x[1] not in EXCLUDED_SOURCES, theme_matches))
  seen_titles = set()
  unique_matches = []
  for entry in theme_matches:
    if entry[2] not in seen_titles:
      seen_titles.add(entry[2])
      unique_matches.append(entry)

  theme_matches = unique_matches
  theme_matches.sort(reverse=True)  # Sort descending by score
  if not theme_matches:
    print("No matches found for the specified themes.")
    return
  print(f"Found {len(theme_matches)} papers matching the themes.")
  print("Top matching titles:")
  for score, source, title in theme_matches[:5]:
      print(f"[{score}] {source} - {title}")
  print("...")


  print("Saving sorted theme matches to file...")
  os.makedirs("outputs", exist_ok=True)
  output_file = os.path.join(base_path, "outputs", output_file) if output_file else os.path.join(base_path, "outputs", "theme_matches.txt")
  with open(output_file, "w", encoding="utf-8") as f:
      for score, source, title in theme_matches:
          f.write(f"[{score}] {source} - {title}\n")
  print(f"Sorted theme matches saved to {output_file}.")


if __name__ == "__main__":
  PATH = "C:/Users/mathi/Desktop/Malta/Code/MALTA-TTS/Paper_finding"
  OUTPUT = "theme_matches.txt"     
  EXCLUDED_SOURCES = ("theme_matches", "output_", "log_") #("icassp2024", "icassp2025")
  WEIGHTED_PATTERNS = [
    # General themes
    (r"\bmaltese\b", 100),
    (r"\btts\b|\btext-to-speech\b|\bspeech synthesis\b|\bmultilingual TTS\b|\bzero[- ]shot TTS\b|\bcross-lingual TTS\b", 80),
    (r"\blow[- ]resource\b|\bunder[- ]resourced\b|\blow[- ]resource languages\b|\blanguages with limited resources\b", 95),
    (r"\bdataset\b.*(maltese|low[- ]resource|under[- ]resourced)", 90),
    (r"\bdata scarcity\b|\blow data availability\b", 88),

    # Specific themes
    (r"\btransfer learning\b|\blanguage transfer\b", 87),
    (r"\bmultilingual\b|\bcross[- ]lingual\b|\bcode-switching\b", 73),
    (r"\bspeech foundation model\b|\baudio[- ]language model\b", 60),
    (r"\bfoundational model(s)?\b", 58),
    (r"\bfew[- ]shot\b|\blimited data\b", 65),
    (r"\bzero[- ]shot\b", 48),

    # Improving models
    (r"\b(domain|task|language) adaptation\b|\b(domain|task|language) extension\b", 86),
    (r"\bself[- ]supervised\b|\bself[- ]training\b|\bpseudo[- ]labeling\b", 35),
    (r"\binstruction tuning\b|\bprompt tuning\b", 30),
    # (r"\btraining data\b|\btraining efficiency\b", 7),
    # (r"\bdata efficiency\b|\blabel efficiency\b", 3),

    # Models
    (r"\bwhisper\b", 80),
    (r"\blo[- ]rank adaptation\b|\bLoRA\b", 70),
    (r"\bdiffusion (model|transformer|TTS)\b", 50),

    # Text processing
    (r"\btext[- ]to[- ]phoneme\b", 85),
    (r"\btext normalization\b", 85),
    (r"\bgrapheme[- ]to[- ]phoneme\b|\bG2P\b", 85),
    (r"\btext preprocessing\b", 75),
    (r"\blinguistic pre[- ]processing\b", 75),
    (r"\bTTS alignment\b|\btext[- ]to[- ]speech alignment\b", 75),
    (r"\btext analysis pipeline\b", 70),
    (r"\bpart[- ]of[- ]speech tagging\b|\bPOS tagging\b", 70),
    (r"\bsyntactic parsing\b", 70),
    (r"\butterance segmentation\b", 70),
    (r"\btokenization\b", 65),
    (r"\bsubword segmentation\b", 65),
    (r"\bsemantic parsing\b", 65),
    (r"\bdependency parsing\b", 65),
    (r"\bbyte[- ]pair encoding\b|\bBPE\b", 60),
    (r"\bwordpiece\b", 60),
    (r"\bsentencepiece\b", 60),
    (r"\blatent representation(s)?\b", 60),
    (r"\bconstituency parsing\b", 60),
    (r"\bnamed entity recognition\b|\bNER\b", 60),
    (r"\bcontextual embeddings\b", 55),
    (r"\bpretrained language model(s)?\b", 55),
    (r"\btoken encoder(s)?\b", 50),
    (r"\btext encoder(s)?\b", 50),
    (r"\btext representation(s)?\b", 50),
    (r"\btext feature extraction\b", 50),
    (r"\bfront[-]end(s)?\b", 45),
    (r"\blabeling\b", 30),
    
    # Useful techniques for later
    (r"\bgeneralization\b", 65),
    (r"\brobust( training| models| generalization)?\b", 64),
    (r"\bdata augmentation\b", 55),
    (r"\bcompositional generalization\b", 23),
    (r"\bbenchmark\b", 40),
    (r"\bontology\b", 12),
  ]

  find_and_score_titles(PATH, WEIGHTED_PATTERNS, OUTPUT, EXCLUDED_SOURCES)
    