import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def paper_count(df):
  st.markdown("---")
  st.markdown("### 📊 Paper Count by Score Range")

  # Define score bins
  bin_size = 10
  min_score = int(df['score'].min())
  max_score = int(df['score'].max())
  bins = range(min_score - (min_score % bin_size), max_score + bin_size, bin_size)
  df['score_bin'] = pd.cut(df['score'], bins=bins, right=False, include_lowest=True,
                        labels=[f"{i}-{i+bin_size-1}" for i in bins[:-1]])

  score_counts_binned = df['score_bin'].value_counts().sort_index()

  # Remove 0-count bins
  score_counts_binned = score_counts_binned[score_counts_binned > 0]
  span = score_counts_binned.max() / score_counts_binned.min() if not score_counts_binned.empty else 1
  use_log_scale = span >= 20

  # Plot
  fig2, ax2 = plt.subplots(figsize=(10, 6))
  ax2.bar(score_counts_binned.index.astype(str), score_counts_binned.values, width=0.8)

  scale_label = "Log" if use_log_scale else "Linear"
  ax2.set_title(f"Count of Papers per Score Range (Bin Size: {bin_size}) [{scale_label} Scale]")
  ax2.set_xlabel("Score Range")
  ax2.set_ylabel(f"Number of Papers")
  if use_log_scale:
    ax2.set_yscale('log')
  ax2.grid(True, axis="y", linestyle='--', alpha=0.7)

  if len(score_counts_binned) > 15:
    plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  st.pyplot(fig2)