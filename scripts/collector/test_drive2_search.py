import requests
import re

s = requests.Session()
s.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
})

queries = ["квадратный номер", "номер тип 1а", "квадратные номера", "японские номера", "гост 1а"]
for q in queries:
    url = f"https://www.drive2.ru/search/?text={requests.utils.quote(q)}"
    r = s.get(url, timeout=5)
    print(f"Query: {q} -> status {r.status_code}")
    if r.status_code == 200:
        posts = re.findall(r'href="(/l/\d+/|/b/\d+/|/o/\d+/)"', r.text)
        imgs = re.findall(r'https://[a-zA-Z0-9.-]+\.drive-data\.ru/[a-zA-Z0-9_-]+-(?:480|960|1920)\.jpg', r.text)
        print(f"   posts: {len(posts)}, direct imgs on search page: {len(imgs)}")
