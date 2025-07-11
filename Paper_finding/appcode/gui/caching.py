import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer

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
