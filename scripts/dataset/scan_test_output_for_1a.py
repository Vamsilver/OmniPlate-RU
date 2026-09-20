import os
from pathlib import Path

dirs = [
    'pure_1a_quota', 'new_1a_candidates', 'new_1a_verified',
    'deep_audit_type1a', 'drive2_test_candidates', 'pure_1a_fresh'
]

for d in dirs:
    p = Path('test_output') / d
    if p.exists():
        jpgs = list(p.glob('*.jpg'))
        jsons = list(p.glob('*.json'))
        print(f"{d}: {len(jpgs)} jpgs, {len(jsons)} jsons")
