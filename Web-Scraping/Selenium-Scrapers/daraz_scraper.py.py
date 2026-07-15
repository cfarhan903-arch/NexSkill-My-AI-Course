from selenium import webdriver
from selenium.webdriver.common.by import By
import csv
import time

url = "https://www.daraz.pk/catalog/?spm=a2a0e.tm80331704.cate_5.5.77cc5aa7fPImi7&q=Smart%20Phones&from=hp_categories&src=all_channel"
driver = webdriver.Chrome()
driver.get(url)

print("Scrolling to load products...")

for _ in range(8):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

productsdiv = driver.find_elements(By.CLASS_NAME, "Bm3ON")
products=[]

print(f"Found {len(productsdiv)} products. Extracting URLs...")

for product in productsdiv:

    try:
        p_url = product.find_element(By.TAG_NAME, "a").get_attribute("href")
        p_url = p_url.split("?")[0] if p_url else "N/A"
    except:
        p_url = "N/A"

    try:
        p_price = product.find_element(By.CLASS_NAME, "ooOxS").text
    except:
        p_price = "N/A"

    try:
        p_name = product.find_element(By.CLASS_NAME, "RfADt").text
    except:
        p_name = "N/A"

    try:
        units_sold = product.find_element(By.CLASS_NAME, "c3e8SH").text
    except:
        units_sold = "N/A"

    print(p_name, p_price, units_sold)

    products.append({
        "Product Name": p_name,
        "Price": p_price,
        "Product URL": p_url,
        "Units Sold": units_sold
    })

with open('products.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['Product Name', 'Price', 'Product URL', 'Units Sold'])
    writer.writeheader()
    writer.writerows(products)  
print("Data saved to products.csv")
driver.quit()