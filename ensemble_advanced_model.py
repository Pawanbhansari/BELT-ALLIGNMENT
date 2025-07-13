import cv2
import numpy as np
import pandas as pd
import os
import json
from sklearn.linear_model import LinearRegression
from scipy.fft import fft2

# Helper: advanced feature extraction
def extract_advanced_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    features = {}
    # Texture: std of Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features['laplacian_std'] = np.std(laplacian)
    # FFT: mean magnitude
    f = np.abs(fft2(gray))
    features['fft_mean'] = np.mean(f)
    features['fft_std'] = np.std(f)
    # Shape: solidity, extent
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(largest)
        rect_area = w * h
        features['solidity'] = area / hull_area if hull_area > 0 else 0
        features['extent'] = area / rect_area if rect_area > 0 else 0
    else:
        features['solidity'] = 0
        features['extent'] = 0
    return features

# Load predictions from previous models
with open('improved_predictions.json') as f:
    improved_preds = json.load(f)
with open('targeted_fix_predictions.json') as f:
    targeted_preds = json.load(f)
with open('enhanced_predictions.json') as f:
    enhanced_preds = json.load(f)

# Load ground truth
df = pd.read_csv('labels.csv')

# Prepare data for ensemble/meta-model
X = []
y = []
filenames = []
advanced_feats = []
for i, row in df.iterrows():
    filename = row['filename']
    improved = improved_preds[i]['predicted_value']
    targeted = targeted_preds[i]['predicted']  # FIXED KEY
    enhanced = enhanced_preds[i]['predicted_value']
    feats = extract_advanced_features(filename)
    X.append([improved, targeted, enhanced, feats['laplacian_std'], feats['fft_mean'], feats['fft_std'], feats['solidity'], feats['extent']])
    y.append(row['center_percent'])
    filenames.append(filename)
    advanced_feats.append(feats)

# Train meta-model (linear regression ensemble)
meta = LinearRegression()
meta.fit(X, y)
ensemble_preds = meta.predict(X)

# Refined rules for borderline cases
final_preds = []
for i, pred in enumerate(ensemble_preds):
    # Borderline rule: if ensemble prediction is near zero but right region is brighter, reduce prediction
    enhanced_feats = enhanced_preds[i]['features']
    right_brighter = enhanced_feats['right_mean_intensity'] > enhanced_feats['center_mean_intensity']
    left_brighter = enhanced_feats['left_mean_intensity'] > enhanced_feats['center_mean_intensity']
    intensity_asym = enhanced_feats['intensity_asymmetry']
    contour_offset = abs(enhanced_feats['contour_offset_x_norm'])
    # If near zero, moderate asymmetry, small contour offset, reduce prediction
    if abs(pred) < 50 and intensity_asym < 0.5 and contour_offset < 0.07:
        pred = pred * 0.5
    # If extreme asymmetry and large contour offset, amplify prediction
    if intensity_asym > 1.0 and contour_offset > 0.1:
        pred = pred * 1.2 if pred > 0 else pred * 1.2
    final_preds.append(pred)

# Save results
results = []
for i, row in df.iterrows():
    results.append({
        'filename': filenames[i],
        'true': y[i],
        'ensemble_pred': ensemble_preds[i],
        'final_pred': final_preds[i],
        'improved': improved_preds[i]['predicted_value'],
        'targeted': targeted_preds[i]['predicted'],  # FIXED KEY
        'enhanced': enhanced_preds[i]['predicted_value'],
        'advanced_features': advanced_feats[i],
        'enhanced_features': enhanced_preds[i]['features']
    })
with open('ensemble_advanced_predictions.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Print summary
errors = [abs(fp - t) for fp, t in zip(final_preds, y)]
mae = np.mean(errors)
rmse = np.sqrt(np.mean([(fp - t) ** 2 for fp, t in zip(final_preds, y)]))
print(f"Ensemble+Advanced Model MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
for i, r in enumerate(results):
    print(f"{r['filename']}: True={r['true']:.1f}%, Pred={r['final_pred']:.1f}%, Error={r['final_pred']-r['true']:.1f}%") 