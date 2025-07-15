import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st

def paper_count(df, score_range=None):
  st.markdown("---")
  st.markdown("### 📊 Paper Count by Score Range")

  # Define score bins
  bin_size = 10
  min_score = int(df["score"].min())
  max_score = int(df["score"].max())

  bins = range(min_score - (min_score % bin_size), max_score + bin_size * 2, bin_size)
  bin_labels = [f"{i}-{i+bin_size-1}" for i in bins[:-1]]

  df['score_bin'] = pd.cut(df['score'], bins=bins, right=False, include_lowest=True, labels=bin_labels)
  score_counts_binned = df['score_bin'].value_counts().sort_index()

  # Remove 0-count bins
  score_counts_binned = score_counts_binned[score_counts_binned > 0]

  span = score_counts_binned.max() / score_counts_binned.min() if not score_counts_binned.empty else 1
  use_log_scale = span >= 20

  fig, ax = plt.subplots(figsize=(12, 6))

  bars = ax.bar(score_counts_binned.index.astype(str), score_counts_binned.values,
                color="#4682B4", width=0.8, edgecolor='white')

  # Highlight selected range as a single background rectangle behind bars
  if score_range:
    start, end = score_range
    highlight_labels = [label for label in bin_labels
                        if int(label.split('-')[1]) >= start and int(label.split('-')[0]) <= end]

    xticks = list(score_counts_binned.index.astype(str))
    indices = [xticks.index(label) for label in highlight_labels if label in xticks]
    if indices:
      left = min(indices) - 0.4
      right = max(indices) + 0.4
      ax.axvspan(left, right, color="orange", alpha=0.25, zorder=0)

  # Axis and style
  ax.set_title(f"Count of Papers per Score Range (Bin Size: {bin_size})", fontsize=14)
  ax.set_xlabel("Score Range", fontsize=12)
  ax.set_ylabel("Number of Papers", fontsize=12)
  ax.grid(True, axis="y", linestyle="--", alpha=0.5)

  if use_log_scale:
    ax.set_yscale("log")
    ax.set_ylabel("Number of Papers (log scale)", fontsize=12)

  # Ticks
  plt.xticks(rotation=45, ha='right')
  ax.tick_params(axis='x', labelsize=9)
  ax.tick_params(axis='y', labelsize=10)

  # Count labels toggle
  show_labels = st.checkbox("🔢 Show counts on bars", value=False)
  if show_labels:
    for bar in bars:
      height = bar.get_height()
      if height > 0:
          ax.annotate(f"{int(height)}",
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha="center", va="bottom", fontsize=8, color="black")

  plt.tight_layout()
  st.pyplot(fig)
