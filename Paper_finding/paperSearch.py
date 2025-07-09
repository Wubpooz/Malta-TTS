import re
import os

weighted_patterns = [
  # r"\blanguages\b"
  (r"\bmaltese\b", 100),
  (r"\btts\b|\btext-to-speech\b|\bspeech synthesis\b|\bmultilingual TTS\b|\bzero[- ]shot TTS\b|\bcross-lingual TTS\b", 80),
  (r"\blow[- ]resource\b|\bunder[- ]resourced\b|\blow[- ]resource languages\b|\blanguages with limited resources\b", 95),
  (r"\btransfer learning\b|\blanguage transfer\b", 78),
  (r"\bmultilingual\b|\bcross[- ]lingual\b|\bcode-switching\b", 65),
  (r"\bbenchmark\b", 40),
  (r"\binstruction tuning\b|\bprompt tuning\b", 20),
  (r"\b(domain|task|language) adaptation\b|\b(domain|task|language) extension\b", 63),
  (r"\bself[- ]supervised\b|\bself[- ]training\b|\bpseudo[- ]labeling\b", 15),
  (r"\bspeech foundation model\b|\baudio[- ]language model\b", 58),
  (r"\bfoundational model(s)?\b", 58),
  (r"\bwhisper\b", 80),
  (r"\blo[- ]rank adaptation\b|\bLoRA\b", 70),
  (r"\bdiffusion (model|transformer|TTS)\b", 50),
  (r"\bdataset\b.*(maltese|low[- ]resource|under[- ]resourced)", 90),
  (r"\bzero[- ]shot\b", 55),
  (r"\bontology\b", 12),
  (r"\bdata augmentation\b", 35),
  (r"\bcompositional generalization\b", 23),
  (r"\btraining data\b|\btraining efficiency\b", 7),
  (r"\bgeneralization\b", 10),
  (r"\bdata efficiency\b|\blabel efficiency\b", 3),
  (r"\bfew[- ]shot\b|\blimited data\b", 76),
  (r"\brobust( training| models| generalization)?\b", 44),
  (r"\bdata scarcity\b|\blow data availability\b", 82),
]
EXCLUDE_PREFIXES = ("theme_matches", "output_", "log_")


def find_and_score_titles(base_path, weighted_patterns, output_file=None):
  txt_files = [f for f in os.listdir(base_path) if f.endswith(".txt") and not f.startswith(EXCLUDE_PREFIXES)]
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

  theme_matches = []
  for source, titles in papers_by_file.items():
    for title in titles:
      score = sum(weight for pattern, weight in weighted_patterns if pattern.search(title))
      if score > 0:
        theme_matches.append((score, source, title))
  theme_matches.sort(reverse=True)  # Sort descending by score


  print(f"Found {len(theme_matches)} papers matching the themes.")
  for score, source, title in theme_matches[:10]:
      print(f"[{score}] {source} - {title}")
  print("...")


  # Save the matches to a text file
  os.makedirs("outputs", exist_ok=True)
  output_file = os.path.join(base_path, "outputs", output_file) if output_file else os.path.join(base_path, "outputs", "theme_matches.txt")
  with open(output_file, "w", encoding="utf-8") as f:
      for score, source, title in theme_matches:
          f.write(f"[{score}] {source} - {title}\n")
  print(f"Sorted theme matches saved to {output_file}.")


if __name__ == "__main__":
    compiled_weighted_patterns = [(re.compile(pat, re.IGNORECASE), weight) for pat, weight in weighted_patterns]
    find_and_score_titles("C:/Users/mathi/Desktop/Malta/Code/MALTA-TTS/Paper_finding", compiled_weighted_patterns, "theme_matches.txt")
    