import requests
import re
import time
from concurrent.futures import ThreadPoolExecutor

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
}

cars = [
    '/r/toyota/probox/805895/',
    '/r/toyota/probox/484306285806747792/',
    '/r/toyota/probox/476605306365805124/',
    '/r/toyota/probox/585678509108655174/',
    '/r/toyota/probox/711671821413390027/',
    '/r/toyota/succeed/684476775689879934/',
    '/r/toyota/succeed/687865470526686418/',
    '/r/toyota/succeed/748905018/',
    '/r/toyota/wish/484306285806747792/',
    '/r/toyota/wish/585678509108655174/',
]

t0 = time.time()
total_imgs = 0
s = requests.Session()
s.headers.update(headers)

for c in cars:
    r = s.get(f"https://www.drive2.ru{c}", timeout=5)
    imgs = re.findall(r'https://[a-zA-Z0-9.-]+\.drive-data\.ru/[a-zA-Z0-9_-]+-(?:480|960|1920)\.jpg', r.text)
    total_imgs += len(imgs)
    time.sleep(0.2)

dt = time.time() - t0
print(f"Paced 1 thread: {len(cars)} cars in {dt:.2f}s -> {total_imgs} photos ({total_imgs/dt:.1f} photos/s)")
