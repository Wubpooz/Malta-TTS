import re
import os

theme_patterns = [
    r"\bmaltese\b",
    r"\btts\b|\btext-to-speech\b|\bspeech synthesis\b|\bmultilingual TTS\b|\bzero[- ]shot TTS\b|\bcross-lingual TTS\b",
    r"\bvoice cloning\b|\bspeaker adaptation\b|\bmultispeaker\b|\bspeaker embedding\b",
    r"\blow[- ]resource\b|\bunder[- ]resourced\b",
    r"\bbenchmark",
    r"\blanguage transfer\b|\btransfer learning\b",
    r"\bmultilingual\b|\bcross[- ]lingual\b|\bcode-switching\b",
    r"\binstruction tuning\b|\bprompt tuning\b",
    r"\b(domain|task|language) adaptation\b|\b(domain|task|language) extension\b",
    r"\bself[- ]supervised\b|\bself[- ]training\b|\bpseudo[- ]labeling\b",
    r"\bspeech foundation model\b|\baudio[- ]language model\b",
    r"\bfoundational model(s)?\b",
    r"\bwhisper\b",
    r"\blo[- ]rank adaptation\b|\bLoRA\b",
    r"\bdiffusion (model|transformer|TTS)\b",
    r"\bLLM(s)? for (tts|text-to-speech|low[- ]resource|under[- ]resourced)",
    r"\bdataset\b.*(maltese|low[- ]resource|under[- ]resourced)",
    r"\bzero[- ]shot\b",
    r"\bontology\b",
    r"\bdata augmentation\b",
    r"\bcompositional generalization\b",
    r"\btraining data\b|\btraining efficiency\b",
    r"\bgeneralization\b",
    r"\bdata efficiency\b|\blabel efficiency\b",
    r"\bfew[- ]shot\b|\blimited data\b",
    r"\brobust( training| models| generalization)?\b",
    r"\btransfer learning\b",
    r"\bdata scarcity\b|\blow data availability\b"
]


compiled_patterns = [re.compile(pat, flags=re.IGNORECASE) for pat in theme_patterns]


base_path = "C:/Users/mathi/Desktop/Malta/Code/MALTA-TTS/Paper_finding"

txt_files = [f for f in os.listdir(base_path) if f.endswith(".txt")]
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


theme_matches = []
for source, titles in papers_by_file.items():
    for title in titles:
        if any(pat.search(title) for pat in compiled_patterns):
            theme_matches.append((source, title))

print(f"Found {len(theme_matches)} papers matching the themes.")
for match in theme_matches[:10]:
    print(match)
print("...")


# Save the matches to a text file
output_file = os.path.join(base_path, "theme_matches.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for source, title in theme_matches:
        f.write(f"{source}: {title}\n")
print(f"Theme matches saved to {output_file}.")

