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
