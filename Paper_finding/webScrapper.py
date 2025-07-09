from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

# Initialize browser
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
# Optional: Suppress non-critical log messages
options.add_experimental_option('excludeSwitches', ['enable-logging'])

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

all_titles = []

for page in range(1, 28):
    url = f"https://ieeexplore.ieee.org/xpl/conhome/10445798/proceeding?sortType=paper-citations&isnumber=10445803&rowsPerPage=100&pageNumber={page}"
    driver.get(url)
    wait = WebDriverWait(driver, 10)

    try:
        # --- NEW: Look for the cookie banner and click "Accept All" ---
        # This only needs to run if the banner appears (usually only on the first page).
        try:
            # The cookie banner is inside a shadow DOM, but we can find the host first
            # and then click the button. A more direct approach is finding the button by its ID.
            cookie_button = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
            cookie_button.click()
            print("Cookie banner accepted.")
        except Exception:
            # If the button is not found after 10 seconds, we assume it's not there and continue.
            print("Cookie banner not found, proceeding with scraping.")

        # Wait for the title elements to be present
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h2.result-item-title a")))
        
        titles = driver.find_elements(By.CSS_SELECTOR, "h2.result-item-title a")
        
        for t in titles:
            title_text = t.text.strip()
            if title_text:
                all_titles.append(title_text)
        
        print(f"Page {page} scraped, found {len(titles)} titles. Total so far: {len(all_titles)}")

    except Exception as e:
        print(f"Error on page {page}: {e}")
        # Take a screenshot to help debug what the page looks like when it fails
        driver.save_screenshot(f"error_page_{page}.png")


driver.quit()

# Save to file
with open("ieee_titles_selenium.txt", "w", encoding="utf-8") as f:
    for title in all_titles:
        f.write(title + "\n")

print(f"✅ Done! {len(all_titles)} titles saved to ieee_titles_selenium.txt")