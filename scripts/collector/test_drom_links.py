import requests
import re

r = requests.get('https://auto.drom.ru/region25/page1/', headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
links = set(re.findall(r'href="([^"]+)"', r.text))
print("Total unique links:", len(links))
for l in sorted(links):
    if any(k in l for k in ['/toyota/', '/nissan/', '/honda/']) and any(c.isdigit() for c in l):
        print(l)
