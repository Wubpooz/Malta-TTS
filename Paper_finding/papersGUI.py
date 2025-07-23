import streamlit as st
import os

# import functions
from paperSearch import find_and_score_titles
from paperSearch_values import PATH, OUTPUT, EXCLUDED_SOURCES, WEIGHTED_PATTERNS, boost_patterns, CATEGORIES, research_goal, model_name, scoring_type
from appcode.gui.caching import load_matches

# import gui elements
from appcode.gui.footer import footer_gui
from appcode.gui.inputs import inputs, cleanup_user_uploads
from appcode.gui.paperListAndFiltering import paper_list_and_filtering
from appcode.gui.graphics import paper_count


# ======================== Page Setup ========================
st.set_page_config(page_title="Paper Relevance Dashboard", layout="wide")
footer_gui()
st.title("📚 Paper Relevance Dashboard")

# Error handling
if not os.path.exists(PATH):
  st.error(f"📁 Folder not found: {PATH}. Please make sure it exists.")
  st.stop()
cleanup_user_uploads(PATH)

# File input and model
[base_path,
  match_file,
  weighted_patterns,
  boost_patterns,
  CATEGORIES,
  research_goal,
  model_name,
  scoring_mode_internal,
  use_semantic,
  semantic_threshold] = inputs(PATH, OUTPUT, WEIGHTED_PATTERNS, boost_patterns, CATEGORIES, research_goal, model_name, scoring_type)


if st.button("🔄 Run Scoring"):
  with st.spinner("🔄 Scoring in progress..."):
    find_and_score_titles(
      base_path=base_path,
      output_file=os.path.basename(match_file),
      EXCLUDE_SOURCES=EXCLUDED_SOURCES,
      weighted_patterns=[(p, w) for p, w, _ in weighted_patterns], # Pass only Pattern and Weight
      boost_patterns=[(p, w) for p, w, _ in boost_patterns], 
      categorize=True,
      CATEGORIES=CATEGORIES,
      research_goal=research_goal,
      model_name="all-MiniLM-L6-v2",
      scoring_type=scoring_mode_internal,
      semantic_threshold= semantic_threshold if use_semantic else 0.3
    )
  st.session_state.reset_needed = True
  st.success("✅ Scoring complete!")


# Load and display
if os.path.exists(match_file):
  df = load_matches(match_file)
  # st.success(f"Loaded {len(df)} scored papers from {match_file}")

  selected_df = paper_list_and_filtering(df, use_semantic, research_goal)

  paper_count(df, score_range=st.session_state.get("last_score_range", None))
else:
  st.warning(f"No file found at: {match_file}")
