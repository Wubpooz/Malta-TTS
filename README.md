# Explanation
GUI version:  
`pip install -r requirements.txt`

Just run `streamlit run Paper_findings/papersGUI.py`

&nbsp;  
CLI:  
Change your params in <paperSearch_values.py> then run:`py paperSearch.py`  

&nbsp;  
Docker deployment :  
`docker build -t paper-dashboard .`
`docker run -p 8501:8501 paper-dashboard`


Structure should be:
Paper_findings
  ├── paperSearch.py
  ├── paperSearch_values.py
  ├── papersGUI.py
  ├── requirements.txt
  ├── code (containes other python functions)
  ├── sources (**put your sources here, as .txt**)
  └── outputs
Dockerfile