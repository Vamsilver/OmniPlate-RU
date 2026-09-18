import requests

r = requests.get('https://www.drive2.ru/robots.txt')
print("Robots status:", r.status_code)
lines = [line for line in r.text.splitlines() if 'sitemap' in line.lower()]
print("Sitemaps:", lines[:10])
