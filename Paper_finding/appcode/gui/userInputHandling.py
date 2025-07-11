import os
import re
import time
from pathlib import Path
import streamlit as st

def upload_papers_gui(base_path):
  st.markdown("### 📤 Upload Additional `.txt` Papers")

  uploaded_files = st.file_uploader(
    "Upload your .txt papers to be included in scoring:",
    type=["txt"],
    accept_multiple_files=True
  )

  user_upload_dir = Path(base_path) / "user_uploads"
  os.makedirs(user_upload_dir, exist_ok=True)

  MAX_FILE_SIZE_MB = 2
  uploaded_file_paths = []

  for uploaded_file in uploaded_files:
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
      st.warning(f"{uploaded_file.name} is too large. Max size: {MAX_FILE_SIZE_MB} MB.")
      continue

    try:
      # 1. Try to decode to ensure it's safe text
      content = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
      st.error(f"{uploaded_file.name} is not valid UTF-8 text.")
      continue

    # 2. Sanitize filename
    safe_filename = re.sub(r"[^a-zA-Z0-9_\-.]", "_", uploaded_file.name)
    target_path = user_upload_dir / safe_filename

    # 3. Prevent overwriting critical paths (only allow user_uploads folder)
    if not str(target_path.resolve()).startswith(str(user_upload_dir.resolve())):
      st.error(f"Unsafe filename detected: {uploaded_file.name}")
      continue

    # 4. Save file to user_uploads/
    with open(target_path, "w", encoding="utf-8") as f:
      f.write(content)
    uploaded_file_paths.append(str(target_path))

  if uploaded_file_paths:
    st.success(f"Uploaded {len(uploaded_file_paths)} text file(s) to: `{user_upload_dir}`")



def cleanup_user_uploads(directory, max_age_seconds=86400):  # 24 hours
  now = time.time()
  for file in Path(directory).glob("*.txt"):
    if file.is_file() and now - file.stat().st_mtime > max_age_seconds:
      try:
        file.unlink()
      except Exception as e:
        st.warning(f"Failed to delete {file.name}: {e}")
