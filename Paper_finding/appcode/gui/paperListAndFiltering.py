import streamlit as st
import pandas as pd
import os
from sentence_transformers import util
from appcode.gui.caching import load_model, load_goal_embedding, compute_embeddings

def paper_list_and_filtering(df, use_semantic, research_goal):
  st.markdown("---")
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


  # Save original filtered dataframe in session_state to track checkboxes
  if "paper_df" not in st.session_state or st.session_state.get("reset_needed", False):
      st.session_state.paper_df = filtered_df.copy()
      st.session_state.reset_needed = False
  else:
      # update the filtered_df from session_state to preserve 'selected'
      filtered_df = st.session_state.paper_df.copy()


  if "selected" not in filtered_df.columns:
    filtered_df["selected"] = False


  st.markdown(f"### Showing {len(filtered_df)} filtered results")
  edited_df = st.data_editor(
      filtered_df,
      use_container_width=True,
      num_rows="dynamic",
      column_config={
          "tags": st.column_config.TextColumn("Tags"),
          "selected": st.column_config.CheckboxColumn("Select")
      }
  )
  
  st.session_state.paper_df = edited_df.copy()

  edited_df = edited_df.sort_values(by="score", ascending=False)

  selected_df = edited_df[edited_df["selected"] == True]


  col1, col2 = st.columns(2)
  with col1:
    if st.button("💾 Save selected to notes"):
      os.makedirs("outputs", exist_ok=True)
      with open("outputs/selected_notes.md", "a", encoding="utf-8") as f:
        for _, row in selected_df.iterrows():
          f.write(f"- **{row.title}**\n  Score: {row.score}, Source: {row.source}\n  Tags: {row.tags}\n\n")
      st.success("Saved to notes!")
  with col2:
    if st.button("📤 Export table to CSV"):
      export_path = "outputs/exported_papers.csv"
      edited_df.to_csv(export_path, index=False)
      st.success(f"Exported to {export_path}")