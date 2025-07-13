import json
import pandas as pd
import numpy as np

BORDERLINE_MIN = 20
BORDERLINE_MAX = 50

# Load ground truth
labels = pd.read_csv('labels.csv')
with open('enhanced_predictions.json', 'r') as f:
    enhanced_preds = json.load(f)

borderline_cases = []
print("Borderline cases (20% < |error| < 50%):\n")
for i, row in labels.iterrows():
    true_val = row['center_percent']
    pred_val = enhanced_preds[i]['predicted_value']
    error = pred_val - true_val
    abs_error = abs(error)
    if BORDERLINE_MIN < abs_error < BORDERLINE_MAX:
        borderline_cases.append({
            'filename': row['filename'],
            'true': true_val,
            'pred': pred_val,
            'error': error,
            'features': enhanced_preds[i].get('features', {})
        })
        print(f"{row['filename']}: True={true_val:.1f}%, Pred={pred_val:.1f}%, Error={error:.1f}%")

print(f"\nTotal borderline cases: {len(borderline_cases)}\n")

# Analyze features of borderline cases
for case in borderline_cases:
    print(f"\n--- {case['filename']} ---")
    print(f"True: {case['true']:.1f}%, Pred: {case['pred']:.1f}%, Error: {case['error']:.1f}%")
    features = case['features']
    for k, v in features.items():
        print(f"{k}: {v}") 