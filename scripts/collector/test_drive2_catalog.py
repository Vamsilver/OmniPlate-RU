import requests
import re
import time

s = requests.Session()
s.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
})

models = ['toyota/probox', 'toyota/succeed', 'toyota/wish', 'toyota/allion', 'nissan/wingroad', 'subaru/forester']
total_cars = set()
for m in models:
    for p in range(1, 3):
        url = f"https://www.drive2.ru/cars/{m}/?page={p}"
        r = s.get(url, timeout=5)
        print(f"{url} -> {r.status_code}")
        if r.status_code == 200:
            cars = re.findall(r'href="(/r/[a-zA-Z0-9_/-]+)"', r.text)
            cars = [c for c in cars if c.count('/') >= 4]
            total_cars.update(cars)
            print(f"   found {len(cars)} cars, total unique {len(total_cars)}")
        time.sleep(0.35)
