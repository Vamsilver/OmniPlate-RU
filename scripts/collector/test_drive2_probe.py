import requests
import re

s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
})

r = s.get("https://www.drive2.ru/cars/toyota/probox/?page=1")
print("Status:", r.status_code, "Len:", len(r.text))
car_links = set(re.findall(r'/r/[a-zA-Z0-9_/-]+', r.text))
car_links = [l for l in car_links if l.count('/') >= 4]
print("Car links found:", len(car_links), car_links[:5])

if car_links:
    first_car = f"https://www.drive2.ru{car_links[0]}"
    rc = s.get(first_car)
    print("First car status:", rc.status_code, "Len:", len(rc.text))
    imgs = re.findall(r'https://[a-zA-Z0-9.-]+\.drive-data\.ru/[a-zA-Z0-9_-]+-(?:480|960|1920)\.jpg', rc.text)
    print("Images found:", len(imgs), imgs[:3])
