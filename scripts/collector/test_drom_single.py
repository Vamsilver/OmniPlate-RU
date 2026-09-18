import requests
import re

url = "https://auto.drom.ru/vladivostok/toyota/succeed/748905018.html"
r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
print("Car status:", r.status_code)
# Find photo URLs
photo_dirs = set(re.findall(r'https://[a-zA-Z0-9.-]+\.drom\.ru/photo/v2/[a-zA-Z0-9_-]+/', r.text))
print("Found photo sets:", len(photo_dirs))
imgs = [ps + "gen1200.jpg" for ps in photo_dirs]
print("Gen1200 imgs:", len(imgs), imgs[:2])
