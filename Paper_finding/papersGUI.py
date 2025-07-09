import streamlit as st
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import ast
import re
from paperSearch import find_and_score_titles, WEIGHTED_PATTERNS, boost_patterns, CATEGORIES, EXCLUDED_SOURCES, OUTPUT, PATH, research_goal


st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px 0;
        font-size: 0.875rem;
        z-index: 100;
        border-top: 1px solid #ccc;
    }

    @media (prefers-color-scheme: light) {
        .footer {
            background-color: #f0f2f6;
            color: #333;
            border-color: #e0e0e0;
        }
    }

    @media (prefers-color-scheme: dark) {
        .footer {
            background-color: #0e1117;
            color: #ccc;
            border-color: #222;
        }
        .footer a {
            color: #61dafb;
        }
    }

    .footer a {
        text-decoration: none;
    }
    </style>

    <div class="footer">
        © 2025 Mathieu Waharte — <a href="https://github.com/Wubpooz/Malta-TTS" target="_blank">View on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)



# ========================= Helper Functions =========================
# --- Pattern editing via DataFrame interface ---
def tuple_list_to_df(tuple_list):
  return pd.DataFrame(tuple_list, columns=["Pattern", "Weight"])

def df_to_tuple_list(df):
  return [(row["Pattern"], int(row["Weight"])) for _, row in df.iterrows() if row["Pattern"].strip()]

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



# ========================= Cacheing Functions =========================
@st.cache_resource(show_spinner=False)
def load_model(name="all-MiniLM-L6-v2"):
  return SentenceTransformer(name)

@st.cache_data(show_spinner=False)
def load_goal_embedding(_model, goal):
  return _model.encode(goal, convert_to_tensor=True)

@st.cache_data(show_spinner=False)
def compute_embeddings(_model, titles):
  return _model.encode(titles, convert_to_tensor=True)

def load_matches(path):
  with open(path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip() and line.startswith("[")]
  data = []
  for line in lines:
    try:
      score = float(line.split("]")[0][1:])
      rest = line.split("]")[1].strip()
      source, title = rest.split(" - ", 1)
      data.append({"score": score, "source": source, "title": title, "tags": ""})
    except:
      continue
  return pd.DataFrame(data)




# ======================== Page Setup ========================
st.set_page_config(page_title="Paper Relevance Dashboard", layout="wide")
st.title("📚 Paper Relevance Dashboard")

# File input and model
match_file = st.text_input("📁 Path to matches file:", os.path.join(PATH, "outputs", OUTPUT))
base_path = st.text_input("📂 Path to paper files:", PATH)
research_goal = st.text_input("🎯 Research goal:", research_goal)
scoring_mode = st.selectbox("🔢 Scoring type:", ["patterns (fast, regex-based)", "semantic (slow, often underselects)", "combined (very slow, experimental)"], index=0)
scoring_label_to_value = {
    "patterns (fast, regex-based)": "patterns",
    "semantic (EXPERIMENTAL, slow, often underselects)": "semantic",
    "combined (EXPERIMENTAL, very slow)": "combined"
}
scoring_mode_internal = scoring_label_to_value[scoring_mode]

use_semantic = st.checkbox("Use semantic similarity in filter", value=False)
semantic_threshold = st.slider("Semantic similarity threshold:", 0.0, 1.0, 0.5, 0.01) if use_semantic else None



st.markdown("### 📝 Weighted Patterns")
wp_df = st.data_editor(tuple_list_to_df(WEIGHTED_PATTERNS), num_rows="dynamic", key="wp_editor")
if st.button("🧠 Convert weighted patterns to regex"):
  wp_df = convert_column_to_regex(wp_df)
  st.success("Converted to regex-friendly format!")

st.markdown("### 🚀 Boost Patterns")
bp_df = st.data_editor(tuple_list_to_df(boost_patterns), num_rows="dynamic", key="bp_editor")
if st.button("🧠 Convert boost patterns to regex"):
  bp_df = convert_column_to_regex(bp_df)
  st.success("Converted to regex-friendly format!")

# Final parsed pattern tuples
weighted_patterns = df_to_tuple_list(wp_df)
boost_patterns = df_to_tuple_list(bp_df)


# Automatically run scoring
with st.spinner("🔄 Scoring in progress..."):
  find_and_score_titles(
    base_path=base_path,
    output_file=os.path.basename(match_file),
    EXCLUDE_SOURCES=EXCLUDED_SOURCES,
    weighted_patterns=weighted_patterns,
    boost_patterns=boost_patterns,
    categorize=True,
    CATEGORIES=CATEGORIES,
    research_goal=research_goal,
    model_name="all-MiniLM-L6-v2",
    scoring_type=scoring_mode_internal,
    semantic_threshold= semantic_threshold if use_semantic else None
  )

# Load and display
if os.path.exists(match_file):
  df = load_matches(match_file)
  st.success(f"Loaded {len(df)} scored papers from {match_file}")

  st.markdown("### 📑 Paper List and Filtering")

  # Score filter
  score_range = st.slider("Filter by score:", 0.0, float(df['score'].max()), (50.0, float(df['score'].max())))
  keyword_filter = st.text_input("🔍 Filter by keyword (optional)").lower()

  filtered_df = df[(df["score"] >= score_range[0]) & (df["score"] <= score_range[1])]
  if keyword_filter:
    filtered_df = filtered_df[filtered_df["title"].str.lower().str.contains(keyword_filter)]

  if use_semantic and research_goal.strip():
    model = load_model()
    goal_embedding = load_goal_embedding(model, research_goal)
    title_embeddings = compute_embeddings(model, filtered_df["title"].tolist())
    similarities = util.cos_sim(goal_embedding, title_embeddings)[0].tolist()
    filtered_df["semantic_score"] = similarities
    filtered_df.sort_values("semantic_score", ascending=False, inplace=True)

  #TODO Manual tagging
  # st.markdown("### 🏷️ Tag papers")
  # for idx, row in filtered_df.iterrows():
  #     new_tag = st.text_input(f"Tags for: {row['title'][:50]}...", value=row["tags"], key=f"tag_{idx}")
  #     df.at[idx, "tags"] = new_tag

  # Selection and saving
  selected = st.multiselect("✅ Select papers to save:", filtered_df["title"].tolist())


  col1, col2 = st.columns(2)
  with col1:
    if st.button("💾 Save selected to notes"):
      os.makedirs("outputs", exist_ok=True)
      with open("outputs/selected_notes.md", "a", encoding="utf-8") as f:
        for title in selected:
          row = filtered_df[filtered_df["title"] == title].iloc[0]
          f.write(f"- **{row.title}**\n  Score: {row.score}, Source: {row.source}\n  Tags: {row.tags}\n\n")
      st.success("Saved to notes!")
  with col2:
    if st.button("📤 Export table to CSV"):
      export_path = "outputs/exported_papers.csv"
      filtered_df.to_csv(export_path, index=False)
      st.success(f"Exported to {export_path}")


  st.markdown(f"### Showing {len(filtered_df)} filtered results")
  filtered_df = filtered_df.sort_values(by="score", ascending=False)
  st.dataframe(filtered_df.reset_index(drop=True))

  
  # Number of papers per score (rounded to 0.1) and plot
  st.markdown("### 📊 Paper Count by Rounded Score")
  rounded_scores = df["score"].round(10)
  score_counts = rounded_scores.value_counts().sort_index()

  fig2, ax2 = plt.subplots(figsize=(4, 2))  # 🔽 Reduced size here
  ax2.bar(score_counts.index.astype(str), score_counts.values, width=0.4)
  ax2.set_title("📏 Count of Papers per Rounded Score to the nearest integer")
  ax2.set_xlabel("Rounded Score")
  ax2.set_ylabel("Number of Papers")
  ax2.grid(True, axis="y")
  plt.xticks(rotation=45)
  st.pyplot(fig2)
else:
  st.warning(f"No file found at: {match_file}")
