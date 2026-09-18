import requests
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Sec-Ch-Ua': '"Chromium";v="128", "Not;A=Brand";v="24"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
}

s = requests.Session()
s.headers.update(headers)
r1 = s.get('https://www.drive2.ru/cars/toyota/probox/')
print('Probox:', r1.status_code)
time.sleep(1)
r2 = s.get('https://www.drive2.ru/cars/toyota/wish/')
print('Wish:', r2.status_code)
