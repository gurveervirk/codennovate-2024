from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType
from bs4 import BeautifulSoup
import time

def get_driver(options):
    return webdriver.Chrome(
        service=Service(
            ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
        ),
        options=options,
    )

def scrape_urls(base_url, is_streamlit=False):
    if is_streamlit:
        options = Options()
        options.add_argument("--disable-gpu")
        options.add_argument("--headless")
        driver = get_driver(options=options)
    else:
        driver = webdriver.Chrome()
    driver.get(base_url)
    all_links = []
    current_page = 1

    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'pagination')))
        
        def get_page_links(retries=3):
            for _ in range(retries):
                try:
                    page_links = driver.find_elements(By.CSS_SELECTOR, '.pagination li a')
                    return [link for link in page_links if link.text.strip()]
                except StaleElementReferenceException:
                    time.sleep(1)
            return []

        page_links = get_page_links()
        max_pages = int([link for link in page_links if link.text.strip() not in ['Next', 'Prev', '...']][-1].text)
        
        while current_page <= max_pages:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'table')))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            all_links.extend([link['href'] for link in soup.find_all('a', href=True) if link['href'].endswith('.pdf')])
            
            if current_page == max_pages:
                break
            
            page_links = get_page_links()
            next_buttons = [link for link in page_links if link.text == 'Next' and 'disabled' not in link.get_attribute('class')]
            
            if not next_buttons:
                break
            
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            next_buttons[0].click()
            current_page += 1
            time.sleep(2)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'pagination')))
    
    except (TimeoutException, NoSuchElementException) as e:
        print(f"An error occurred: {e}")
    
    finally:
        driver.quit()
    
    return all_links
