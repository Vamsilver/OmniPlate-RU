import requests
import re
import time
import cv2
import numpy as np

s = requests.Session()
s.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
})

r = s.get('https://www.drive2.ru/cars/toyota/bb/?page=1', timeout=5)
print('bB page 1 status:', r.status_code)
cars = re.findall(r'href="(/r/toyota/bb/[^"]+)"', r.text)
print('Found bB cars:', len(cars), cars[:3])

if cars:
    car_url = 'https://www.drive2.ru' + cars[0]
    r_car = s.get(car_url, timeout=5)
    print('First car status:', r_car.status_code)
    imgs = re.findall(r'https://[a-zA-Z0-9.-]+\.drive-data\.ru/[a-zA-Z0-9_-]+-1920\.jpg', r_car.text)
    print('Found 1920px photos:', len(imgs), imgs[:2])
