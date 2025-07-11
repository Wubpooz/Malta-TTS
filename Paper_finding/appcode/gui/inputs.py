import streamlit as st
import os
from appcode.gui.helperFunctions import tuple_list_to_df, df_to_tuple_list, convert_column_to_regex
from appcode.gui.userInputHandling import upload_papers_gui

def inputs(PATH, OUTPUT, WEIGHTED_PATTERNS, boost_patterns, CATEGORIES, research_goal, model_name, scoring_type):
  # Paths
  match_file = st.text_input("📁 Path to matches file:", os.path.join(PATH, "outputs", OUTPUT))
  sources_path = st.text_input("📂 Path to paper files:", os.path.join(PATH, "sources"))
  upload_papers_gui(base_path=sources_path)

  # Patterns
  st.markdown("### 📝 Weighted Patterns")
  wp_df = st.data_editor(
      tuple_list_to_df(WEIGHTED_PATTERNS),
      num_rows="dynamic",
      key="wp_editor",
      column_config={
          "Enabled": st.column_config.CheckboxColumn("Enabled", default=True),
          "Pattern": st.column_config.TextColumn("Pattern"),
          "Weight": st.column_config.NumberColumn("Weight")
      }
  )
  if st.button("🧠 Convert weighted patterns to regex", key="wp_regex_btn"):
      wp_df = convert_column_to_regex(wp_df)
      st.success("Converted to regex-friendly format!")

  st.markdown("### 🚀 Boost Patterns")
  bp_df = st.data_editor(
      tuple_list_to_df(boost_patterns),
      num_rows="dynamic",
      key="bp_editor",
      column_config={
          "Enabled": st.column_config.CheckboxColumn("Enabled", default=True),
          "Pattern": st.column_config.TextColumn("Pattern"),
          "Weight": st.column_config.NumberColumn("Weight")
      }
  )
  if st.button("🧠 Convert boost patterns to regex", key="bp_regex_btn"):
      bp_df = convert_column_to_regex(bp_df)
      st.success("Converted to regex-friendly format!")

  weighted_patterns = df_to_tuple_list(wp_df)
  boost_patterns = df_to_tuple_list(bp_df)

  # Scoring type
  scoring_label_to_value = {
      "patterns (fast, regex-based)": "patterns",
      "semantic (EXPERIMENTAL, slow, often underselects)": "semantic",
      "combined (EXPERIMENTAL, very slow)": "combined"
  }
  index_of_scoring_mode = list(scoring_label_to_value.values()).index(scoring_type) if scoring_type in scoring_label_to_value.values() else 0
  scoring_mode = st.selectbox("🔢 Scoring type:", list(scoring_label_to_value.keys()), index=index_of_scoring_mode)
  scoring_mode_internal = scoring_label_to_value[scoring_mode]

  # Semantic
  use_semantic = scoring_mode_internal == "semantic" or scoring_mode_internal == "combined"
  if use_semantic:
    st.markdown(f"#### Semantic Scoring Parameters")
  research_goal = st.text_input("🎯 Research goal:", research_goal) if use_semantic else None
  model_name = st.text_input("🧠 SentenceTransformer model:", model_name) if use_semantic else None
  semantic_threshold = st.slider("Semantic similarity threshold:", 0.0, 1.0, 0.5, 0.01) if use_semantic else None

  return sources_path, match_file, weighted_patterns, boost_patterns, CATEGORIES, research_goal, model_name, scoring_mode_internal, use_semantic, semantic_threshold