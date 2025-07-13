import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_performance():
    """Analyze the segmentation model performance"""
    
    # Load ground truth
    df = pd.read_csv('labels.csv')
    
    # Load predictions
    with open('segmentation_predictions.json', 'r') as f:
        predictions = json.load(f)
    
    # Filter to only belt images (exclude PNG files)
    belt_predictions = [p for p in predictions if p['filename'].endswith('.jpg')]
    
    # Match predictions with ground truth
    results = []
    for pred in belt_predictions:
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
                'belt_area': pred['belt_area']
            })
    
    results_df = pd.DataFrame(results)
    
    print("=== SEGMENTATION MODEL PERFORMANCE ANALYSIS ===\n")
    
    # Basic statistics
    print(f"Total samples: {len(results_df)}")
    print(f"Mean Absolute Error (MAE): {results_df['abs_error'].mean():.2f}")
    print(f"Root Mean Square Error (RMSE): {np.sqrt((results_df['error']**2).mean()):.2f}")
    print(f"Standard Deviation of Error: {results_df['error'].std():.2f}")
    
    # Calculate R-squared
    ss_res = np.sum(results_df['error'] ** 2)
    ss_tot = np.sum((results_df['true'] - results_df['true'].mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"R-squared (R²): {r2:.4f}")
    
    print("\n=== DETAILED COMPARISON ===")
    for _, row in results_df.iterrows():
        print(f"{row['filename']}: True={row['true']:.1f}%, Predicted={row['predicted']:.1f}%, Error={row['error']:.1f}%")
    
    # Analyze error patterns
    print(f"\n=== ERROR ANALYSIS ===")
    print(f"Largest overestimation: {results_df['error'].max():.1f}%")
    print(f"Largest underestimation: {results_df['error'].min():.1f}%")
    print(f"Average belt area: {results_df['belt_area'].mean():.0f} pixels")
    
    # Check if errors correlate with belt area
    correlation = results_df['abs_error'].corr(results_df['belt_area'])
    print(f"Correlation between error and belt area: {correlation:.3f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Scatter plot
    plt.subplot(2, 3, 1)
    plt.scatter(results_df['true'], results_df['predicted'], alpha=0.7, s=100)
    plt.plot([results_df['true'].min(), results_df['true'].max()], 
            [results_df['true'].min(), results_df['true'].max()], 'r--', lw=2)
    plt.xlabel('True Alignment (%)')
    plt.ylabel('Predicted Alignment (%)')
    plt.title('Predictions vs True Values')
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(2, 3, 2)
    plt.hist(results_df['error'], bins=8, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Prediction Error (%)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Error vs True value
    plt.subplot(2, 3, 3)
    plt.scatter(results_df['true'], results_df['error'], alpha=0.7, s=100)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Alignment (%)')
    plt.ylabel('Prediction Error (%)')
    plt.title('Error vs True Value')
    plt.grid(True, alpha=0.3)
    
    # Belt area vs error
    plt.subplot(2, 3, 4)
    plt.scatter(results_df['belt_area'], results_df['abs_error'], alpha=0.7, s=100)
    plt.xlabel('Belt Area (pixels)')
    plt.ylabel('Absolute Error (%)')
    plt.title('Error vs Belt Area')
    plt.grid(True, alpha=0.3)
    
    # Sample-wise errors
    plt.subplot(2, 3, 5)
    x_pos = range(len(results_df))
    plt.bar(x_pos, results_df['error'], alpha=0.7, color='skyblue')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Error (%)')
    plt.title('Error by Sample')
    plt.xticks(x_pos, [f"{i+1}" for i in range(len(results_df))])
    plt.grid(True, alpha=0.3)
    
    # True vs Predicted comparison
    plt.subplot(2, 3, 6)
    x_pos = range(len(results_df))
    width = 0.35
    plt.bar([x - width/2 for x in x_pos], results_df['true'], width, label='True', alpha=0.7)
    plt.bar([x + width/2 for x in x_pos], results_df['predicted'], width, label='Predicted', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Alignment (%)')
    plt.title('True vs Predicted Comparison')
    plt.legend()
    plt.xticks(x_pos, [f"{i+1}" for i in range(len(results_df))])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Identify specific issues
    print(f"\n=== SPECIFIC ISSUES IDENTIFIED ===")
    
    # Check for systematic bias
    mean_error = results_df['error'].mean()
    if abs(mean_error) > 10:
        print(f"Systematic bias detected: Model tends to predict {mean_error:.1f}% lower than true values")
    
    # Check for scale issues
    true_range = results_df['true'].max() - results_df['true'].min()
    pred_range = results_df['predicted'].max() - results_df['predicted'].min()
    if pred_range < true_range * 0.5:
        print(f"Scale compression: Predicted range ({pred_range:.1f}) is much smaller than true range ({true_range:.1f})")
    
    # Check for extreme errors
    extreme_errors = results_df[results_df['abs_error'] > 50]
    if not extreme_errors.empty:
        print(f"Extreme errors (>50%) found in {len(extreme_errors)} samples:")
        for _, row in extreme_errors.iterrows():
            print(f"  {row['filename']}: Error = {row['error']:.1f}%")
    
    return results_df

if __name__ == "__main__":
    analyze_performance() 