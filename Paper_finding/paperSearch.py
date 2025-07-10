import re
import os
from sentence_transformers import SentenceTransformer, util

def loading_papers_files(base_path, exclude_sources):
  txt_files = [f for f in os.listdir(base_path) if f.endswith(".txt") and not f.startswith(exclude_sources)]
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
  return papers_by_file



def scoring(scoring_type, papers_by_file, compiled_weighted_patterns, exclude_sources, compiled_boost_patterns=None, model_name="all-MiniLM-L6-v2", research_goal=None, semantic_threshold=0.3):
  """
  Scores paper titles based on compiled patterns and semantic matching.
  Args:
      scoring_type (str): The type of scoring to use. Options are 'patterns', 'semantic', or 'combined'.
      papers_by_file (dict): A dictionary where keys are file names and values are lists of paper titles.
      compiled_weighted_patterns (list): A list of tuples containing compiled regex patterns and their weights.
      exclude_sources (tuple): A tuple of file name prefixes to exclude from processing.
      compiled_boost_patterns (list, optional): A list of tuples containing compiled regex patterns and their bonus scores.
      model_name (str, optional): The name of the SentenceTransformer model to use for semantic scoring.
      research_goal (str, optional): The research goal to use for semantic scoring.
      semantic_threshold (float, optional): The threshold for semantic similarity. Defaults to 0.3.
  Returns:
      list: A list of tuples containing the score, source file name, and title of matched papers.
  """
  SCORING_TYPES = ['patterns', 'semantic', 'combined']
  theme_matches = []
  if scoring_type not in SCORING_TYPES:
    raise ValueError(f"Invalid scoring type. Choose from {SCORING_TYPES}.")
  if not papers_by_file:
    print("No papers found. Exiting scoring.")
    return theme_matches
  if not compiled_weighted_patterns:
    print("No weighted patterns provided. Exiting scoring.")
    return theme_matches

  match scoring_type:
    case 'patterns':
      print("Scoring using patterns...")
      for source, titles in papers_by_file.items():
        for title in titles:
          score = sum(weight for pattern, weight in compiled_weighted_patterns if pattern.search(title))
          for pattern, bonus in compiled_boost_patterns:
            if pattern.search(title):
              score += bonus
          if score > 0:
            theme_matches.append((score, source, title))
        print(f"Found {len(theme_matches)} matches before filtering excluded sources and removing duplicates.")

    
    case 'semantic':
      print("Scoring using semantic matching...")
      if not research_goal:
        raise ValueError("Research goal must be provided for semantic matching.")
      if not model_name:
        raise ValueError("Model name must be provided for semantic matching.")

      model = SentenceTransformer(model_name)
      # Encode goal once
      goal_embedding = model.encode(research_goal, convert_to_tensor=True)

      for source, titles in papers_by_file.items():
        title_embeddings = model.encode(titles, convert_to_tensor=True)
        similarities = util.cos_sim(goal_embedding, title_embeddings)[0]
        for score, title in zip(similarities.tolist(), titles):
            if score > semantic_threshold:
                theme_matches.append((score, source, title))
      print(f"Found {len(theme_matches)} semantically matched papers.")
    
    case 'combined':
      print("Scoring using combined patterns and semantic matching...")
      if not research_goal:
        raise ValueError("Research goal must be provided for combined scoring.")
      if not model_name:
        raise ValueError("Model name must be provided for combined scoring.")

      model = SentenceTransformer(model_name)
      goal_embedding = model.encode(research_goal, convert_to_tensor=True)

      for source, titles in papers_by_file.items():
          title_embeddings = model.encode(titles, convert_to_tensor=True)
          similarities = 100 * util.cos_sim(goal_embedding, title_embeddings)[0]
          for idx, (title, similarity_score) in enumerate(zip(titles, similarities.tolist())):
              pattern_score = sum(weight for pattern, weight in compiled_weighted_patterns if pattern.search(title))
              for pattern, bonus in compiled_boost_patterns:
                  if pattern.search(title):
                      pattern_score += bonus
              if pattern_score == 0 and similarity_score < semantic_threshold:
                  continue
              combined_score = 0.7 * pattern_score + 0.3 * similarity_score  # normalize similarity
              if combined_score > 0:
                  theme_matches.append((combined_score, source, title))
      print(f"Found {len(theme_matches)} matches using combined scoring.")

  theme_matches = list(filter(lambda x: x[1] not in exclude_sources, theme_matches))
  seen_titles = set()
  unique_matches = []
  for entry in theme_matches:
    if entry[2] not in seen_titles:
      seen_titles.add(entry[2])
      unique_matches.append(entry)

  theme_matches = unique_matches
  theme_matches.sort(reverse=True)
  return theme_matches


def categorize_by_score(matches, categories=None):
  categorized_matches = {label: [] for label in categories.keys()} if categories else {}
  for match in matches:
    score = match[0]
    title = match[2]
    source = match[1]
    if categories:
      for label, threshold in categories.items():
        if score >= threshold:
          categorized_matches[label].append((score, source, title))
          break
    else:
      categorized_matches['uncategorized'].append((score, source, title))
  return categorized_matches


def saving_scores_to_file(theme_matches, output_file, base_path, categorize=True, CATEGORIES=None):
  print("Saving sorted theme matches to file...")
  os.makedirs("outputs", exist_ok=True)
  output_file = os.path.join(base_path, "outputs", output_file) if output_file else os.path.join(base_path, "outputs", "theme_matches.txt")
  if os.path.exists(output_file):
    print(f"Output file {output_file} already exists. Overwriting.")
  else:
    print(f"Creating output file {output_file}.")
  
  if categorize:
    categories = CATEGORIES if CATEGORIES else {"high": 200, "medium": 80, "low": 10}
    categorized_matches = categorize_by_score(theme_matches, categories)
    with open(output_file, "w", encoding="utf-8") as f:
      for label in categories.keys():
        f.write(f"\n===== {label.upper()} RELEVANCE =====\n")
        for score, source, title in categorized_matches[label]:
          f.write(f"[{score}] {source} - {title}\n")
  else:
    with open(output_file, "w", encoding="utf-8") as f:
      for score, source, title in theme_matches:
        f.write(f"[{score}] {source} - {title}\n")
    print(f"Sorted theme matches saved to {output_file}.")



def find_and_score_titles(base_path, weighted_patterns, boost_patterns=None, output_file=None, EXCLUDE_SOURCES = ("theme_matches", "output_", "log_"), categorize=True, CATEGORIES=None
                          , scoring_type='patterns', research_goal=None, model_name=None, semantic_threshold=0.3):
  """
  Finds and scores paper titles based on weighted patterns from text files in a specified directory.

  Args:
    base_path (str): The path to the directory containing text files with paper titles.
    weighted_patterns (list of tuples): A list of tuples where each tuple contains a regex pattern 
                                        and its associated weight for scoring.
    boost_patterns (list of tuples, optional): A list of tuples where each tuple contains a regex pattern
                                              and a bonus score to boost the score of titles that match these patterns.
    output_file (str, optional): The name of the file to save the scored paper titles. Defaults to 'theme_matches.txt'.
    EXCLUDE_SOURCES (tuple, optional): A tuple of file name prefixes to exclude from processing. Defaults to 
                                        ("theme_matches", "output_", "log_").
    categorize (bool, optional): Whether to categorize the scores into predefined categories. Defaults to True.
    CATEGORIES (dict, optional): A dictionary defining categories and their score thresholds for categorization
                                        (e.g., {"high": 200, "medium": 80, "low": 10}).
    scoring_type (str, optional): The type of scoring to use. Options are 'patterns', 'semantic', or 'combined'.
                                  Defaults to 'patterns'.
    research_goal (str, optional): The research goal to use for semantic scoring. Required if scoring_type is 'semantic' or 'combined'.
    model_name (str, optional): The name of the SentenceTransformer model to use for semantic scoring. Required if scoring_type is 'semantic' or 'combined'.
    semantic_threshold (float, optional): The threshold for semantic similarity. Defaults to 0.3.
      
  Returns:
      None. Outputs the results by printing and saving scored titles to a text file.

  This function reads text files from the specified directory, applies regex patterns to the titles to compute scores,
  filters out titles from excluded sources, removes duplicates while retaining the highest score, sorts the titles 
  by score in descending order, and saves the results to a file. It also prints the top matching titles and the
  number of matches found.
  """
  papers_by_file = loading_papers_files(base_path, EXCLUDE_SOURCES)

  print(f"Loaded {len(papers_by_file)} files.")
  if not papers_by_file:
    print("No papers found. Exiting.")
    return

  print("Compiling patterns...")
  compiled_weighted_patterns = [(re.compile(pat, re.IGNORECASE), weight) for pat, weight in weighted_patterns]
  if boost_patterns:
    compiled_boost_patterns = [(re.compile(pat, re.IGNORECASE), bonus) for pat, bonus in boost_patterns]

  print("Scoring titles...")
  theme_matches = scoring(scoring_type=scoring_type, papers_by_file=papers_by_file,
                          compiled_weighted_patterns=compiled_weighted_patterns, 
                          compiled_boost_patterns=compiled_boost_patterns,
                          exclude_sources=EXCLUDE_SOURCES,
                          model_name=model_name, research_goal=research_goal, semantic_threshold=semantic_threshold)


  if not theme_matches:
    print("No matches found for the specified themes.")
    return
  print(f"Found {len(theme_matches)} papers matching the themes.")
  print("Top matching titles:")
  for score, source, title in theme_matches[:5]:
    print(f"[{score}] {source} - {title}")
  print("...")


  saving_scores_to_file(theme_matches, output_file, base_path, categorize, CATEGORIES)




from paperSearch_values import PATH, OUTPUT, EXCLUDED_SOURCES, WEIGHTED_PATTERNS, boost_patterns, CATEGORIES, research_goal, model_name, scoring_type

find_and_score_titles(base_path=PATH, output_file= OUTPUT, EXCLUDE_SOURCES=EXCLUDED_SOURCES, 
                      weighted_patterns=WEIGHTED_PATTERNS, boost_patterns=boost_patterns, 
                      categorize=True, CATEGORIES=CATEGORIES, 
                      research_goal=research_goal, model_name=model_name, scoring_type=scoring_type)
