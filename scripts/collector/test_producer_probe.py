import re
import traceback
import requests

s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
})

try:
    cat_url = "https://www.drive2.ru/cars/toyota/probox/?page=1"
    r = s.get(cat_url, timeout=5)
    print("Catalog status:", r.status_code)
    car_links = set(re.findall(r'/r/[a-zA-Z0-9_/-]+', r.text))
    print("Found links:", len(car_links))
    for cl in list(car_links)[:3]:
        print("CL:", cl)
        car_url = f"https://www.drive2.ru{cl}"
        rc = s.get(car_url, timeout=5)
        print("Car status:", rc.status_code)
        imgs = re.findall(r'https://[a-zA-Z0-9.-]+\.drive-data\.ru/[a-zA-Z0-9_-]+-(?:480|960|1920)\.jpg', rc.text)
        print("Found imgs:", len(imgs))
except Exception as e:
    traceback.print_exc()
