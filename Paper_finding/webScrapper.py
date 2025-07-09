from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

# Initialize browser
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # run in background
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

all_titles = []

# There are 27 pages (2700 / 100)
for page in range(1, 28):
    url = f"https://ieeexplore.ieee.org/xpl/conhome/10445798/proceeding?sortType=paper-citations&isnumber=10445803&rowsPerPage=100&pageNumber={page}"
    driver.get(url)

    time.sleep(5)  # wait for JavaScript to load
    
    # Find all paper title elements
    titles = driver.find_elements(By.CLASS_NAME, "title")
    
    for t in titles:
        title_text = t.text.strip()
        if title_text:
            all_titles.append(title_text)
    
    print(f"Page {page} scraped, total titles so far: {len(all_titles)}")

driver.quit()

# Save to file
with open("ieee_titles_selenium.txt", "w", encoding="utf-8") as f:
    for title in all_titles:
        f.write(title + "\n")

print("✅ Done! Titles saved to ieee_titles_selenium.txt")
