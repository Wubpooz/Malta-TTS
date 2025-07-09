#! DOESN'T WORK ON IEEE Xplore due to bot detection
# This script uses Selenium to scrape titles from IEEE Xplore.

import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import requests
import time

# Initialize browser
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
# Optional: Suppress non-critical log messages
options.add_experimental_option('excludeSwitches', ['enable-logging'])

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)



def check_robots_txt(url):
    """
    Checks the robots.txt file for a given URL and prints its content.
    """
    robots_url = f"{url}/robots.txt"
    try:
        response = requests.get(robots_url, timeout=10)
        if response.status_code == 200:
            print(f"Content of {robots_url}:\n")
            print(response.text)
        else:
            print(f"Failed to retrieve robots.txt from {robots_url}. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching robots.txt from {robots_url}: {e}")


ieee_url = "https://ieeexplore.ieee.org"
print(f"Checking robots.txt for {ieee_url}...")
check_robots_txt(ieee_url)

print("\n--- Important Note ---")
print("Even if robots.txt allows access, websites may still employ advanced bot detection.")
print("For academic databases like IEEE Xplore, official APIs or institutional access are the recommended and ethical methods for data retrieval.")



# Use undetected_chromedriver
options = uc.ChromeOptions()
# You can still run headless if you need to
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Initialize the undetected driver
driver = uc.Chrome(options=options)

all_titles = []

print("Driver initialized. Starting scraping...")

for page in range(1, 28):
    url = f"https://ieeexplore.ieee.org/xpl/conhome/10445798/proceeding?sortType=paper-citations&isnumber=10445803&rowsPerPage=100&pageNumber={page}"
    
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 20) # Increased wait time just in case

        # The cookie banner logic can remain, as it's good practice.
        # The undetected driver should get the real page where the banner exists.
        try:
            cookie_button = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
            cookie_button.click()
            print("Cookie banner accepted.")
            # Give the page a moment to settle after clicking the banner
            time.sleep(2)
        except Exception:
            print("Cookie banner not found, proceeding...")

        # Now, wait for the actual content
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h2.result-item-title a")))
        
        titles = driver.find_elements(By.CSS_SELECTOR, "h2.result-item-title a")
        
        for t in titles:
            title_text = t.text.strip()
            if title_text:
                all_titles.append(title_text)
        
        print(f"Page {page} scraped, found {len(titles)} titles. Total so far: {len(all_titles)}")

    except Exception as e:
        print(f"An error occurred on page {page}: {e}")
        driver.save_screenshot(f"error_page_{page}.png")
        # If one page fails, it might be a temporary issue, so we continue to the next
        continue

driver.quit()

# Save to file
with open("ieee_titles_selenium.txt", "w", encoding="utf-8") as f:
    for title in all_titles:
        f.write(title + "\n")

print(f"\n✅ Done! {len(all_titles)} titles saved to ieee_titles_selenium.txt")