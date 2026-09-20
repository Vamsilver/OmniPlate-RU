import json
from pathlib import Path

j = Path('test_output/pure_1a_fresh/candidates_metadata.json')
with open(j, 'r', encoding='utf-8') as f:
    items = json.load(f)

print(f"Total candidates in json: {len(items)}")
for x in items:
    print(f"#{x['id']}: {x['text']} (OCR:{x['ocr_conf']}, Det:{x['conf']:.2f}, AR:{x['ar']}) | {x['crop_fn']}")
