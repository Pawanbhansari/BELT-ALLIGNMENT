import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from improved_model import ImprovedBeltAlignmentModel
import os

def analyze_improved_model():
    # Load ground truth
    df = pd.read_csv('labels.csv')
    
    # Load improved model predictions (regenerate if needed)
    if not os.path.exists('improved_predictions.json'):
        model = ImprovedBeltAlignmentModel()
        model.train('labels.csv', '.')
        results = model.predict_batch('.', 'improved_predictions.json')
    with open('improved_predictions.json', 'r') as f:
        predictions = json.load(f)
    
    # Match predictions with ground truth
    results = []
    for pred in predictions:
        filename = pred['filename']
        gt_row = df[df['filename'] == filename]
        if not gt_row.empty:
            true_value = gt_row['center_percent'].iloc[0]
            pred_value = pred['predicted_value']
            error = pred_value - true_value
            abs_error = abs(error)
            results.append({
                'filename': filename,
                'true': true_value,
                'predicted': pred_value,
                'error': error,
                'abs_error': abs_error,
                'features': pred.get('features', {})
            })
    results_df = pd.DataFrame(results)
    print("=== IMPROVED MODEL DETAILED ERRORS ===\n")
    print(results_df[['filename', 'true', 'predicted', 'error', 'abs_error']])
    print(f"\nMAE: {results_df['abs_error'].mean():.2f}")
    print(f"RMSE: {np.sqrt((results_df['error']**2).mean()):.2f}")
    print(f"R²: {1 - np.sum(results_df['error']**2) / np.sum((results_df['true'] - results_df['true'].mean())**2):.4f}")
    
    # Show worst errors
    print("\n=== LARGEST ERRORS ===")
    print(results_df.sort_values('abs_error', ascending=False).head(3))
    
    # Plot true vs predicted
    plt.figure(figsize=(8,6))
    plt.scatter(results_df['true'], results_df['predicted'], s=100)
    plt.plot([results_df['true'].min(), results_df['true'].max()], [results_df['true'].min(), results_df['true'].max()], 'r--')
    plt.xlabel('True Alignment (%)')
    plt.ylabel('Predicted Alignment (%)')
    plt.title('Improved Model: True vs Predicted')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('improved_model_true_vs_pred.png', dpi=200)
    plt.show()
    
    # Suggest simple calibration if systematic bias
    mean_error = results_df['error'].mean()
    if abs(mean_error) > 10:
        print(f"\nSystematic bias detected: Model predictions are on average {mean_error:.1f}% off.")
        print("Suggesting simple bias correction...")
        results_df['calibrated_predicted'] = results_df['predicted'] - mean_error
        new_mae = np.mean(np.abs(results_df['calibrated_predicted'] - results_df['true']))
        print(f"MAE after bias correction: {new_mae:.2f}")
        if new_mae < results_df['abs_error'].mean():
            print("Bias correction improves results! Apply this correction to future predictions.")
        else:
            print("Bias correction does not improve results.")
    
    # Show features for worst error
    worst = results_df.sort_values('abs_error', ascending=False).iloc[0]
    print(f"\nFeatures for worst error ({worst['filename']}):")
    for k, v in worst['features'].items():
        print(f"  {k}: {v}")
    
    return results_df

if __name__ == "__main__":
    analyze_improved_model() 