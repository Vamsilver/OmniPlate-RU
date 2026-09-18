import requests
import re
import time
from concurrent.futures import ThreadPoolExecutor

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
}

def get_cars_for_model(model):
    s = requests.Session()
    s.headers.update(headers)
    url = f"https://www.drive2.ru/cars/{model}/"
    try:
        r = s.get(url, timeout=5)
        if r.status_code == 200:
            cars = re.findall(r'href="(/r/[a-zA-Z0-9_/-]+)"', r.text)
            return [c for c in cars if c.count('/') >= 4]
    except Exception:
        pass
    return []

test_models = ['toyota/probox', 'toyota/succeed', 'toyota/wish']
all_cars = []
for m in test_models:
    cars = get_cars_for_model(m)
    print(f"{m} -> {len(cars)} cars")
    all_cars.extend(cars)
    time.sleep(0.3)

print("Total cars:", len(all_cars))

def get_car_images(car_path):
    s = requests.Session()
    s.headers.update(headers)
    url = f"https://www.drive2.ru{car_path}"
    for _ in range(3):
        try:
            r = s.get(url, timeout=5)
            if r.status_code == 200:
                imgs = re.findall(r'https://[a-zA-Z0-9.-]+\.drive-data\.ru/[a-zA-Z0-9_-]+-(?:480|960|1920)\.jpg', r.text)
                return [re.sub(r'-(?:480|960)\.jpg', '-1920.jpg', u) for u in set(imgs)]
            elif r.status_code == 429:
                time.sleep(2.0)
        except Exception:
            time.sleep(0.5)
    return []

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(get_car_images, all_cars[:12]))

total_imgs = sum(len(res) for res in results)
print(f"Extracted {total_imgs} high-res photos from 12 cars! Avg: {total_imgs/12:.1f} per car")
