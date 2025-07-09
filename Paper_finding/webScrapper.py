import requests
from bs4 import BeautifulSoup
import time

base_url = "https://ieeexplore.ieee.org/rest/search"
headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0"
}

# Conference ID and other search parameters
payload_template = {
    "newsearch": True,
    "queryText": "",
    "highlight": True,
    "returnFacets": ["ALL"],
    "returnType": "SEARCH",
    "rowsPerPage": 100,
    "pageNumber": 1,
    "refinements": ["ConferenceID:10445798"]
}

all_titles = []

for page in range(1, 28):  # ~2700 / 100 per page = 27 pages
    payload = payload_template.copy()
    payload["pageNumber"] = page

    response = requests.post(base_url, json=payload, headers=headers)
    data = response.json()
    
    records = data.get("records", [])
    for record in records:
        title = record.get("title")
        if title:
            all_titles.append(title)
    
    print(f"Page {page} done, total titles: {len(all_titles)}")
    time.sleep(1)  # be polite

# Save to file
with open("ieee_titles.txt", "w", encoding="utf-8") as f:
    for title in all_titles:
        f.write(title + "\n")
