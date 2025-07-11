import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def paper_count(df):
  st.markdown("---")
  
  # Number of papers per score (rounded to 0.1) and plot
  st.markdown("### 📊 Paper Count by Score Range")

  # Define score bins. Adjust `bin_size` as needed.
  bin_size = 10 
  min_score = int(df['score'].min())
  max_score = int(df['score'].max())

  bins = range(min_score - (min_score % bin_size), max_score + bin_size, bin_size)
  df['score_bin'] = pd.cut(df['score'], bins=bins, right=False, include_lowest=True,
                         labels=[f"{i}-{i+bin_size-1}" for i in bins[:-1]])

  score_counts_binned = df['score_bin'].value_counts().sort_index()

  fig2, ax2 = plt.subplots(figsize=(10, 6))

  ax2.bar(score_counts_binned.index.astype(str), score_counts_binned.values, width=0.8)
  ax2.set_title(f"Count of Papers per Score Range (Bin Size: {bin_size})")
  ax2.set_xlabel("Score Range")
  ax2.set_ylabel("Number of Papers")
  ax2.grid(True, axis="y", linestyle='--', alpha=0.7)

  if len(score_counts_binned) > 15:
    plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  st.pyplot(fig2)