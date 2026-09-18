import requests
import re

r = requests.get('https://auto.drom.ru/region25/page1/', headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
links = re.findall(r'href="([^"]+)"', r.text)
print("Total links:", len(links))
for l in links[:20]:
    if 'auto.drom.ru' in l:
        print(l)
