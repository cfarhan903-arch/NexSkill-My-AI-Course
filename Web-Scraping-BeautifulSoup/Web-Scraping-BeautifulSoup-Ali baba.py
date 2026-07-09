import requests
from bs4 import BeautifulSoup
import csv

URL = "https://www.alibaba.com/trade/search?SearchText=laptop"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

r = requests.get(URL, headers=HEADERS)
soup = BeautifulSoup(r.content, "html.parser")

products = []
items = soup.find_all("div", class_="organic-offer-wrapper")
for item in items:
    product = {}
    try:
        product["Product Name"] = item.find("h2").text.strip()
    except:
        product["Product Name"] = "N/A"
    try:
        product["Price"] = item.find("div", class_="elements-offer-price-normal__price").text.strip()
    except:
        product["Price"] = "N/A"
    try:
        link = item.find("a")["href"]
        if link.startswith("//"):
            link = "https:" + link
        product["Product URL"] = link
    except:
        product["Product URL"] = "N/A"
    products.append(product)

filename = "alibaba_products.csv"
with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["Product Name", "Price", "Product URL"]
    )
    writer.writeheader()
    for product in products:
        writer.writerow(product)

print("Data saved to alibaba_products.csv")