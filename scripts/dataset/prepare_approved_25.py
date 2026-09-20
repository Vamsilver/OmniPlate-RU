import json
from pathlib import Path

j = Path('test_output/pure_1a_fresh/candidates_metadata.json')
with open(j, 'r', encoding='utf-8') as f:
    items = json.load(f)

# Keep only validated 23 items (exclude #5 and #22)
approved = [x for x in items if x['id'] not in [5, 22]]
print(f"Approved from pure_1a_fresh: {len(approved)}")

# Add cand_1a_0008_E328CK125
cand8 = {
    'id': 101,
    'full_fn': '../fresh_1a_harvested/cand_1a_0008_E328CK125.jpg',
    'crop_fn': '../fresh_1a_harvested/cand_8_crop.jpg',
    'url': 'https://www.drive2.ru/r/toyota/probox/125',
    'text': 'E328CK125',
    'conf': 0.9365,
    'ocr_conf': 0.99,
    'bbox': [203, 1079, 120, 105],
    'quad': [203.0, 1079.0, 323.0, 1079.0, 323.0, 1184.0, 203.0, 1184.0],
    'ar': 1.14,
    'num_faces_blurred': 0
}
approved.append(cand8)

# Add cand_1a_0143_E530PO28
with open('test_output/turbo_1a_verified/candidates_metadata.json', 'r', encoding='utf-8') as f:
    turbo = json.load(f)
c143 = next(x for x in turbo if x['full_fn'] == 'cand_1a_0143_E530PO28.jpg')
faces = c143['num_faces_blurred']['faces_detected'] if isinstance(c143['num_faces_blurred'], dict) else 0
approved.append({
    'id': 102,
    'full_fn': f"../turbo_1a_verified/{c143['full_fn']}",
    'crop_fn': f"../turbo_1a_verified/crops/{c143['crop_fn']}",
    'url': c143['url'],
    'text': c143['text'],
    'conf': c143.get('conf_1a', 0.98),
    'ocr_conf': 0.99,
    'bbox': c143['bbox'],
    'quad': c143['quad'],
    'ar': c143['ar'],
    'num_faces_blurred': faces
})

print(f"Total approved for integration: {len(approved)}")
out_path = Path('test_output/pure_1a_fresh/approved_25_candidates.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(approved, f, indent=2, ensure_ascii=False)
print(f"Saved approved list to {out_path}")
