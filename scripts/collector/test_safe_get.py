import requests
import re
import time

s = requests.Session()
s.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
})

def safe_get(url):
    for attempt in range(4):
        try:
            r = s.get(url, timeout=5)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                print(f"[429] Backoff 2.5s on {url}")
                time.sleep(2.5)
                continue
        except Exception:
            time.sleep(1.0)
    return None

# Test 15 requests
urls = [
    'https://www.drive2.ru/cars/toyota/probox/',
    'https://www.drive2.ru/cars/toyota/succeed/',
    'https://www.drive2.ru/cars/toyota/wish/',
    'https://www.drive2.ru/cars/toyota/allion/',
    'https://www.drive2.ru/cars/nissan/wingroad/',
    'https://www.drive2.ru/cars/subaru/forester/',
    'https://www.drive2.ru/cars/honda/fit/',
    'https://www.drive2.ru/cars/mitsubishi/delica/',
]

for u in urls:
    t0 = time.time()
    r = safe_get(u)
    dt = time.time() - t0
    status = r.status_code if r else 'FAILED'
    print(f"{u} -> {status} ({dt:.2f}s)")
    time.sleep(0.4)
