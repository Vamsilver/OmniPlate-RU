import requests
import re

r = requests.get('https://auto.drom.ru/region25/page1/', headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
bulletins = re.findall(r'href="(https://[a-zA-Z0-9.-]+\.drom\.ru/[a-zA-Z0-9_/.-]+)" data-ftid="bulls-list_bull"', r.text)
if not bulletins:
    bulletins = re.findall(r'data-ftid="bulls-list_bull"[^>]*href="([^"]+)"', r.text)
print("Bulletins found:", len(bulletins))
for b in bulletins[:5]:
    print("Bull:", b)
