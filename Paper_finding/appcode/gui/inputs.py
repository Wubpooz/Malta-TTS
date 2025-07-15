import streamlit as st
import os, re, time
from pathlib import Path
import json
from appcode.gui.helperFunctions import tuple_list_to_df, df_to_tuple_list, convert_column_to_regex

def inputs(PATH, OUTPUT, WEIGHTED_PATTERNS, boost_patterns, CATEGORIES, research_goal, model_name, scoring_type):
  # Paths
  match_file = st.text_input("📁 Path to matches file:", os.path.join(PATH, "outputs", OUTPUT))
  sources_base = os.path.join(PATH, "sources")
  sources_path = upload_papers_gui(sources_base)

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

  # Save/download weighted patterns
  if st.download_button("💾 Download weighted patterns", data=json.dumps(df_to_tuple_list(wp_df), indent=2), file_name="weighted_patterns.json", mime="application/json"):
    st.success("Weighted patterns ready for download.")

  # Upload/load weighted patterns
  uploaded_wp_file = st.file_uploader("📤 Upload weighted patterns (JSON)", type="json", key="wp_upload")
  wp_json_input = st.text_area("📋 Or paste weighted patterns (JSON)", key="wp_paste")

  if uploaded_wp_file or wp_json_input:
    try:
      raw = uploaded_wp_file.read().decode() if uploaded_wp_file else wp_json_input
      loaded_wp = json.loads(raw)
      wp_df = tuple_list_to_df(loaded_wp)
      st.success("Weighted patterns loaded!")
    except Exception as e:
      st.error(f"Error loading weighted patterns: {e}")

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

  if st.download_button("💾 Download boost patterns", data=json.dumps(df_to_tuple_list(bp_df), indent=2), file_name="boost_patterns.json", mime="application/json"):
    st.success("Boost patterns ready for download.")

  uploaded_bp_file = st.file_uploader("📤 Upload boost patterns (JSON)", type="json", key="bp_upload")
  bp_json_input = st.text_area("📋 Or paste boost patterns (JSON)", key="bp_paste")

  if uploaded_bp_file or bp_json_input:
    try:
      raw = uploaded_bp_file.read().decode() if uploaded_bp_file else bp_json_input
      loaded_bp = json.loads(raw)
      bp_df = tuple_list_to_df(loaded_bp)
      st.success("Boost patterns loaded!")
    except Exception as e:
      st.error(f"Error loading boost patterns: {e}")

  if st.button("🧠 Convert boost patterns to regex", key="bp_regex_btn"):
    bp_df = convert_column_to_regex(bp_df)
    st.success("Converted to regex-friendly format!")

  # TODO add support for exclude and inverse boost patterns
  # st.markdown("### ❌ Exclude Patterns")
  # exclude_df = st.data_editor(
  #   tuple_list_to_df([]),
  #   num_rows="dynamic",
  #   key="exclude_editor",
  #   column_config={
  #     "Enabled": st.column_config.CheckboxColumn("Enabled", default=True),
  #     "Pattern": st.column_config.TextColumn("Pattern"),
  #     "Weight": st.column_config.NumberColumn("Weight", default=0)
  #   }
  # )
  # if st.button("🧠 Convert exclude patterns to regex", key="excl_regex_btn"):
  #   exclude_df = convert_column_to_regex(exclude_df)
  #   st.success("Exclude patterns converted!")

  # st.markdown("### 🔁 Inverse Boost Patterns")
  # inverse_df = st.data_editor(
  #   tuple_list_to_df([]),
  #   num_rows="dynamic",
  #   key="inverse_editor",
  #   column_config={
  #     "Enabled": st.column_config.CheckboxColumn("Enabled", default=True),
  #     "Pattern": st.column_config.TextColumn("Pattern"),
  #     "Weight": st.column_config.NumberColumn("Weight", default=-1)
  #   }
  # )
  # if st.button("🧠 Convert inverse boost patterns to regex", key="invboost_regex_btn"):
  #   inverse_df = convert_column_to_regex(inverse_df)
  #   st.success("Inverse boost patterns converted!")

  # Final values
  weighted_patterns = df_to_tuple_list(wp_df)
  boost_patterns = df_to_tuple_list(bp_df)
  # exclude_patterns = df_to_tuple_list(exclude_df)
  # inverse_boost_patterns = df_to_tuple_list(inverse_df)

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
  use_semantic = scoring_mode_internal in {"semantic", "combined"}
  if use_semantic:
    st.markdown(f"#### Semantic Scoring Parameters")
  research_goal = st.text_input("🎯 Research goal:", research_goal) if use_semantic else None
  model_name = st.text_input("🧠 SentenceTransformer model:", model_name) if use_semantic else None
  semantic_threshold = st.slider("Semantic similarity threshold:", 0.0, 1.0, 0.5, 0.01) if use_semantic else None

  return (
    sources_path,
    match_file,
    weighted_patterns,
    boost_patterns,
    # exclude_patterns,
    # inverse_boost_patterns,
    CATEGORIES,
    research_goal,
    model_name,
    scoring_mode_internal,
    use_semantic,
    semantic_threshold
  )



def upload_papers_gui(base_path):
  """
  Let user create (or pick) a subfolder under sources/user_sources, 
  then upload .txt files into it.
  """
  st.markdown("### 📤 Upload `.txt` Papers into Your Subfolder")

  # 1. Ask for (or show) existing user subfolders
  root = Path(base_path) / "user_sources"
  os.makedirs(root, exist_ok=True)
  existing = [p.name for p in root.iterdir() if p.is_dir()]

  folder = st.selectbox(
    "Choose or type a folder name:",
    options=[""] + existing,
    index=0,
    help="Pick an existing subfolder or leave blank to type a new one"
  )
  new_folder = st.text_input(
    "New subfolder name (if left blank, reuses above):", 
    value="" if folder else ""
  )
  subfolder = new_folder.strip() or folder or "default"
  user_dir = root / subfolder
  user_dir.mkdir(parents=True, exist_ok=True)
  st.caption(f"Saving into: `{user_dir}`")

  uploaded_files = st.file_uploader(
    "Select one or more `.txt` files to upload:",
    type=["txt"],
    accept_multiple_files=True
  )
  MAX_MB = 2
  saved = 0
  for up in uploaded_files:
    if up.size > MAX_MB * 1024 * 1024:
      st.warning(f"{up.name} ({up.size/1e6:.1f}MB) exceeds {MAX_MB}MB, skipping.")
      continue
    try:
      text = up.getvalue().decode("utf-8")
    except UnicodeDecodeError:
      st.error(f"{up.name} is not valid UTF-8, skipping.")
      continue
    # sanitize filename
    safe = re.sub(r"[^a-zA-Z0-9_\-\.\ ]", "_", up.name)
    tgt = user_dir / safe
    with open(tgt, "w", encoding="utf-8") as f:
      f.write(text)
    saved += 1

  if saved:
    st.success(f"Uploaded {saved} file(s) to `{subfolder}`.")
  else:
    st.warning("No files uploaded.")
  return user_dir

def cleanup_user_uploads(directory, max_age_seconds=86400):  # 24 hours
  now = time.time()
  for file in Path(directory).glob("*.txt"):
    if file.is_file() and now - file.stat().st_mtime > max_age_seconds:
      try:
        file.unlink()
      except Exception as e:
        st.warning(f"Failed to delete {file.name}: {e}")
