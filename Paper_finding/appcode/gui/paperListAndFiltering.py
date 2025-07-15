from io import StringIO
import streamlit as st
from sentence_transformers import util
from appcode.gui.caching import load_model, load_goal_embedding, compute_embeddings

def paper_list_and_filtering(df, use_semantic, research_goal):
  st.markdown("---")
  st.markdown("### 📑 Paper List and Filtering")

  score_range = st.slider("Filter by score:", 0.0, float(df['score'].max()), (50.0, float(df['score'].max())))
  st.session_state.last_score_range = score_range
  keyword_filter = st.text_input("🔍 Filter by keyword (optional)").lower()
  st.session_state.last_keyword_filter = keyword_filter

  filtered_df = df[(df["score"] >= score_range[0]) & (df["score"] <= score_range[1])]
  if keyword_filter:
    filtered_df = filtered_df[filtered_df["title"].str.lower().str.contains(keyword_filter)]

  # TODO Check that code
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
    filtered_df = filtered_df.copy()
    filtered_df["semantic_score"] = st.session_state.semantic_cache[hash_key]
    filtered_df.sort_values("semantic_score", ascending=False, inplace=True)

  # Init persistent selection state
  if "selection_state" not in st.session_state:
    st.session_state.selection_state = {}

  # Select all toggle
  select_all = st.checkbox("✅ Select all in current view")
  if select_all:
    for title in filtered_df["title"]:
      st.session_state.selection_state[title] = True

  # Inject current selection into the view
  filtered_df = filtered_df.copy()
  filtered_df["selected"] = filtered_df["title"].map(lambda title: st.session_state.selection_state.get(title, False))

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

  # Update global state from edited data
  for _, row in edited_df.iterrows():
    st.session_state.selection_state[row["title"]] = row["selected"]

  # Sort here only for display
  sorted_df = edited_df.sort_values(by="score", ascending=False)
  selected_df = sorted_df[sorted_df["selected"] == True]

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
    if not sorted_df.empty:
      csv_buffer = StringIO()
      sorted_df.to_csv(csv_buffer, index=False)
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

  return selected_df
