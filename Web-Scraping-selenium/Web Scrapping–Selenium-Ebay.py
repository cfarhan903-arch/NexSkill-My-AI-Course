from selenium import webdriver
from selenium.webdriver.common.by import By
import csv
import time

driver = webdriver.Chrome()
driver.maximize_window()

url = "https://www.ebay.com/sch/i.html?_nkw=laptop"
driver.get(url)

time.sleep(5)
print("Scrolling to load products...")

for _ in range(8):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

productsdiv = driver.find_elements(By.CSS_SELECTOR, "div[data-qa-locator='product-item']")
products = []

print(f"Found {len(productsdiv)} products.")
for product in products:
    try:
        p_name = product.find_element(By.CLASS_NAME, "RfADt").text
    except:
        p_name = "N/A"
    try:
        p_price = product.find_element(By.CLASS_NAME, "ooOxS").text
    except:
        p_price = "N/A"
    try:
        p_url = product.find_element(By.TAG_NAME, "a").get_attribute("href")
        if p_url:
            p_url = p_url.split("?")[0]
        else:
            p_url = "N/A"
    except:
        p_url = "N/A"
    try:
        units_sold = product.find_element(By.CLASS_NAME, "oa6ri").text
    except:
        units_sold = "N/A"

    print("Product :", p_name)
    print("Price   :", p_price)
    print("Sold    :", units_sold)
    print("URL     :", p_url)

    products.append({
        "Product Name": p_name,
        "Price": p_price,
        "Product URL": p_url,
        "Units Sold": units_sold
    })

# Save CSV
with open("products.csv", "w", newline="", encoding="utf-8") as csvfile:

    fieldnames = [
        "Product Name",
        "Price",
        "Product URL",
        "Units Sold"
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(products)

print("\nData saved successfully to products.csv")
print("Total Products Saved:", len(products))
driver.quit()