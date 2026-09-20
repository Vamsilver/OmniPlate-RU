import requests
import re

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

models = ['toyota/bb', 'toyota/probox', 'toyota/succeed', 'nissan/cube', 'honda/stepwgn']
for m in models:
    url = f"https://www.drive2.ru/cars/{m}/?page=1"
    r = s.get(url, timeout=6)
    pattern = rf'href="(/r/{m}/[0-9]+/)"'
    links = list(set(re.findall(pattern, r.text)))
    print(f"{m}: status={r.status_code}, links={len(links)}")
