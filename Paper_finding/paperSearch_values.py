import os

PATH = os.path.join(os.path.dirname(__file__))
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

# Boost patterns for specific themes associations
boost_patterns = [
  (r"(maltese|low-resource).*(tts|speech synthesis)", 30),
  (r"(text-to-speech).*(alignment|tokenization)", 20),
]


CATEGORIES = {
  "very high": 200,
  "high": 140,
  "medium": 80,
  "low-medium": 50,
  "low": 40,
  "very low": 10,
  "uncategorized": 0
}

research_goal = "low-resource multilingual text-to-speech synthesis using phoneme modeling and robust adaptation"
model_name="all-MiniLM-L6-v2"
scoring_type='patterns'