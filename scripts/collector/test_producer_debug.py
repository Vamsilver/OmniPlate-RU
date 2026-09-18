import re
import requests
import time

s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
})

models = ['toyota/probox', 'toyota/succeed']
for model in models:
    for page in range(1, 3):
        cat_url = f"https://www.drive2.ru/cars/{model}/?page={page}"
        r = s.get(cat_url, timeout=5)
        print(f"Cat {cat_url}: status {r.status_code}")
        car_links = set(re.findall(r'/r/[a-zA-Z0-9_/-]+', r.text))
        print(f"Car links raw: {len(car_links)}")
        valid_links = [l for l in car_links if l.count('/') >= 4]
        print(f"Valid car links: {len(valid_links)}")
        for cl in valid_links[:2]:
            car_url = f"https://www.drive2.ru{cl}"
            rc = s.get(car_url, timeout=5)
            imgs = re.findall(r'https://[a-zA-Z0-9.-]+\.drive-data\.ru/[a-zA-Z0-9_-]+-(?:480|960|1920)\.jpg', rc.text)
            print(f"Car {cl}: imgs {len(imgs)}")
