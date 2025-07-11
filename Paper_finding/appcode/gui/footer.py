import streamlit as st

def footer_gui():
  st.markdown(
      """
      <style>
      .footer {
          position: fixed;
          left: 0;
          bottom: 0;
          width: 100%;
          text-align: center;
          padding: 10px 0;
          font-size: 0.875rem;
          z-index: 100;
          border-top: 1px solid #ccc;
      }

      @media (prefers-color-scheme: light) {
          .footer {
              background-color: #f0f2f6;
              color: #333;
              border-color: #e0e0e0;
          }
      }

      @media (prefers-color-scheme: dark) {
          .footer {
              background-color: #0e1117;
              color: #ccc;
              border-color: #222;
          }
          .footer a {
              color: #61dafb;
          }
      }

      .footer a {
          text-decoration: none;
      }
      </style>

      <div class="footer">
          © 2025 Mathieu Waharte — <a href="https://github.com/Wubpooz/Malta-TTS" target="_blank">View on GitHub</a>
      </div>
      """,
      unsafe_allow_html=True
  )