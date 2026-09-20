import csv
from pathlib import Path

suspects = [
    'real_type1b_0067.jpg',
    'real_type1b_0068.jpg',
    'real_type1b_0084.jpg',
    'real_type1b_0096.jpg',
    'real_type1b_0119.jpg',
    'real_type1b_0123.jpg',
    'real_type1b_0126.jpg',
    'real_type1b_0139.jpg',
    'real_type1b_0141.jpg',
    'real_type1b_0167.jpg',
    'real_type1b_0181.jpg',
    'real_type1b_0183.jpg',
    'real_type1b_0191.jpg',
    'real_type1b_0198.jpg',
    'real_type1b_0208.jpg',
    'real_type1b_0213.jpg',
    'real_type1b_0214.jpg',
    'real_type1b_0232.jpg',
    'real_type1b_0235.jpg',
    'real_type1b_0236.jpg',
    'real_type1b_0237.jpg',
    'real_type1b_0251.jpg',
    'real_type1b_0252.jpg',
    'real_type1b_0266.jpg',
    'real_type1b_0267.jpg',
    'real_type1b_0280.jpg',
    'real_type1b_0284.jpg',
    'real_type1b_0286.jpg',
    'real_type1b_0300.jpg',
    'real_type1b_0302.jpg',
    'real_type1b_0306.jpg',
    'real_type1b_0307.jpg',
    'real_type1b_0308.jpg',
    'real_type1b_0309.jpg',
    'real_type1b_0310.jpg',
]

with open('dataset/meta.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    rows = {Path(r['image']).name: r for r in reader}

for s in suspects:
    r = rows.get(s)
    if r:
        print(f"{s}: plate_num={r.get('plate_num')} | source={r.get('source')} | is_synth={r.get('is_synthetic')}")
    else:
        print(f"{s}: NOT FOUND")
