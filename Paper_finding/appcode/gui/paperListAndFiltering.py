from io import StringIO
import streamlit as st
# import os
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
    titles = filtered_df["title"].tolist()
    hash_key = (tuple(titles), research_goal)

    if "semantic_cache" not in st.session_state:
      st.session_state.semantic_cache = {}

    if hash_key not in st.session_state.semantic_cache:
      model = load_model()
      goal_embedding = load_goal_embedding(model, research_goal)
      title_embeddings = compute_embeddings(model, titles)
      similarities = util.cos_sim(goal_embedding, title_embeddings)[0].tolist()
      st.session_state.semantic_cache[hash_key] = similarities

    filtered_df["semantic_score"] = st.session_state.semantic_cache[hash_key]
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
    if not selected_df.empty:
      notes_md = StringIO()
      for _, row in selected_df.iterrows():
        notes_md.write(f"- **{row.title}**\n  Score: {row.score}, Source: {row.source}\n  Tags: {row.tags}\n\n")
      st.download_button(
        label="💾 Save selected to notes",
        data=notes_md.getvalue(),
        file_name="selected_notes.md",
        mime="text/markdown"
      )
    else:
      st.info("No papers selected.")

  with col2:
    if not edited_df.empty:
      csv_buffer = StringIO()
      edited_df.to_csv(csv_buffer, index=False)
      st.download_button(
        label="📤 Export table to CSV",
        data=csv_buffer.getvalue(),
        file_name="exported_papers.csv",
        mime="text/csv"
      )
    else:
      st.info("No data to export.")

  if not selected_df.empty:
    sel_csv = selected_df.to_csv(index=False).encode('utf-8')
    st.download_button(
      label="📥 Download selected papers",
      data=sel_csv,
      file_name="selected_papers.csv",
      mime="text/csv"
    )
