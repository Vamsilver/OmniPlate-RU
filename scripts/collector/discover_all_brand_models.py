import time
import requests
import re

s = requests.Session()
s.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
})

brands = [
    'toyota', 'nissan', 'honda', 'subaru', 'mitsubishi', 
    'suzuki', 'daihatsu', 'mazda', 'isuzu', 'dodge', 'jeep', 'chrysler'
]

discovered = []
for b in brands:
    time.sleep(0.5)
    url = f"https://www.drive2.ru/cars/{b}/"
    r = s.get(url, timeout=6)
    if r.status_code == 200:
        links = re.findall(rf'href="(/cars/{b}/[a-zA-Z0-9_-]+/m\d+/)"', r.text)
        print(f"Brand {b:12s}: found {len(links)} exact model links")
        discovered.extend(links)

discovered = sorted(list(set(discovered)))
print(f"\n[+] Total unique exact models discovered: {len(discovered)}")
with open("test_output/discovered_models.txt", "w", encoding="utf-8") as f:
    for m in discovered:
        f.write(m + "\n")
