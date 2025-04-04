from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd
import re


options = Options()
options.add_argument("--headless") 
options.add_argument("--disable-gpu")  
options.add_argument("--no-sandbox")  
options.add_argument("--disable-dev-shm-usage")


service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

categories = [ "watches", "laptops", "mobile phones","computers","shoes","headphones","tablets","books"]  

product_data = []

for category in categories:
    print(f"Scraping Category: {category}")

    for page in range(1, 30):  
        url = f"https://www.amazon.in/s?k={category}&page={page}"
        print(f"Scraping Page {page} of {category}...")

        driver.get(url)  
        time.sleep(3)  

        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        products = soup.findAll("div", class_="sg-col-inner")

        for product in products:
            try:
                name_tag = product.find("span", class_="a-size-base-plus a-color-base")
                name = name_tag.get_text(strip=True) if name_tag else "Unknown"

                price_tag = product.find("span", class_="a-price-whole")
                price = price_tag.get_text(strip=True) if price_tag else "Unknown"

                rating_tag = product.find("span", class_="a-icon-alt")  
                rating_text = rating_tag.get_text(strip=True) if rating_tag else "Unknown"
                rating_match = re.search(r"\d+(\.\d+)?", rating_text)  
                rating = rating_match.group() if rating_match else "Unknown"


                reviews_tag = product.find("span", class_="a-size-base s-underline-text")
                reviews = reviews_tag.get_text(strip=True) if reviews_tag else "Unknown"

            
                product_data.append([name, price, category, rating, reviews])

            except Exception as e:
                print(f"Error extracting product: {e}")


driver.quit()

df = pd.DataFrame(product_data, columns=["Name", "Price", "Category", "Rating", "Reviews"])
df.to_csv("amazon_products_.csv", index=False)
print(f"Scraping complete! {len(df)} products")




