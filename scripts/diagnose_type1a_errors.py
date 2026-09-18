import pandas as pd

df_c = pd.read_csv('test_output/omniplate_ground_truth_consolidated.csv', sep=';')
df_p = pd.read_csv('test_output/pipeline_live_predictions.csv', sep=';')
m = df_c.merge(df_p, on='filename')
t1a = m[m['plate_type'] == 'type1a']
errors = t1a[t1a['ground_truth'] != t1a['pred_text']]

print(f"Total Type 1A in consolidated: {len(t1a)}")
print(f"Correct: {len(t1a) - len(errors)}")
print(f"Errors: {len(errors)}")
print("\nFirst 25 Errors:")
for idx, r in errors.head(25).iterrows():
    print(f"- {r.filename:<22} | GT: '{r.ground_truth}' | PRED: '{r.pred_text}' (pred_type: {r.pred_type})")
