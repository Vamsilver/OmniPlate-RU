import requests
from bs4 import BeautifulSoup

r = requests.get('https://auto.drom.ru/region25/page1/', headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
soup = BeautifulSoup(r.text, 'html.parser')
links = [a.get('href') for a in soup.find_all('a') if a.get('href')]
car_links = [l for l in links if 'drom.ru' in l and any(l.rstrip('/').endswith(f"/{i}") or l.rstrip('/').split('/')[-1].isdigit() for i in range(10))]
print("Total links:", len(links))
print("Car candidates:", len(car_links))
for c in car_links[:5]:
    print("Candidate:", c)
