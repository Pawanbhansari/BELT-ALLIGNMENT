import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LinearRegression
from scipy.fft import fft2

def visualize_prediction_process(image_path, image_index=0):
    """
    Visualize the complete prediction process for a specific image
    """
    print(f"=== DETAILED PREDICTION VISUALIZATION FOR {image_path} ===\n")
    
    # Load all model predictions
    with open('improved_predictions.json') as f:
        improved_preds = json.load(f)
    with open('targeted_fix_predictions.json') as f:
        targeted_preds = json.load(f)
    with open('enhanced_predictions.json') as f:
        enhanced_preds = json.load(f)
    with open('ensemble_advanced_predictions.json') as f:
        ensemble_preds = json.load(f)
    
    # Load ground truth
    df = pd.read_csv('labels.csv')
    true_value = df.iloc[image_index]['center_percent']
    
    # Get predictions from each model
    improved_pred = improved_preds[image_index]['predicted_value']
    targeted_pred = targeted_preds[image_index]['predicted']
    enhanced_pred = enhanced_preds[image_index]['predicted_value']
    final_pred = ensemble_preds[image_index]['final_pred']
    
    print("STEP 1: INDIVIDUAL MODEL PREDICTIONS")
    print("=" * 50)
    print(f"Improved Model:      {improved_pred:.1f}%")
    print(f"Targeted Fix Model:  {targeted_pred:.1f}%")
    print(f"Enhanced Model:      {enhanced_pred:.1f}%")
    print(f"True Value:          {true_value:.1f}%")
    print(f"Final Ensemble:      {final_pred:.1f}%")
    
    # Load and process image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    center_x, center_y = width // 2, height // 2
    
    # Extract features for visualization
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Original image with regions
    ax1 = plt.subplot(3, 4, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Draw region boundaries
    cv2.line(img_rgb, (width//3, 0), (width//3, height), (255, 255, 0), 3)
    cv2.line(img_rgb, (2*width//3, 0), (2*width//3, height), (255, 255, 0), 3)
    cv2.circle(img_rgb, (center_x, center_y), 10, (255, 0, 0), -1)
    
    ax1.imshow(img_rgb)
    ax1.set_title('Image with Regions (Yellow lines)\nBlue dot = center')
    ax1.axis('off')
    
    # 2. Grayscale with intensity analysis
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(gray, cmap='gray')
    ax2.set_title('Grayscale Image')
    ax2.axis('off')
    
    # 3. Edge detection
    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(edges, cmap='gray')
    ax3.set_title('Edge Detection')
    ax3.axis('off')
    
    # 4. Contours
    ax4 = plt.subplot(3, 4, 4)
    contour_img = image.copy()
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 3)
        
        # Get contour center
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            contour_center_x = int(M["m10"] / M["m00"])
            contour_center_y = int(M["m01"] / M["m00"])
            cv2.circle(contour_img, (contour_center_x, contour_center_y), 8, (255, 0, 0), -1)
            cv2.circle(contour_img, (center_x, center_y), 8, (0, 0, 255), -1)
    
    ax4.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    ax4.set_title('Contours (Green)\nBlue=Contour center, Red=Image center')
    ax4.axis('off')
    
    # 5. Intensity analysis by region
    ax5 = plt.subplot(3, 4, 5)
    left_region = gray[:, :width//3]
    center_region = gray[:, width//3:2*width//3]
    right_region = gray[:, 2*width//3:]
    
    left_intensity = np.mean(left_region)
    center_intensity = np.mean(center_region)
    right_intensity = np.mean(right_region)
    
    regions = ['Left', 'Center', 'Right']
    intensities = [left_intensity, center_intensity, right_intensity]
    colors = ['red', 'green', 'blue']
    
    bars = ax5.bar(regions, intensities, color=colors, alpha=0.7)
    ax5.set_title('Average Intensity by Region')
    ax5.set_ylabel('Intensity')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, intensities):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{value:.1f}', ha='center', va='bottom')
    
    # 6. Advanced features
    ax6 = plt.subplot(3, 4, 6)
    
    # Texture analysis
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_std = np.std(laplacian)
    
    # FFT analysis
    f = np.abs(fft2(gray))
    fft_mean = np.mean(f)
    
    # Shape analysis
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        hull = cv2.convexHull(largest)
        hull_area = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
    else:
        solidity = 0
    
    advanced_features = {
        'Laplacian Std': laplacian_std,
        'FFT Mean': fft_mean / 1000,  # Scale down for visualization
        'Solidity': solidity
    }
    
    feature_names = list(advanced_features.keys())
    feature_values = list(advanced_features.values())
    
    bars = ax6.bar(feature_names, feature_values, color=['orange', 'purple', 'brown'], alpha=0.7)
    ax6.set_title('Advanced Features')
    ax6.set_ylabel('Value')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, feature_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(feature_values)*0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 7. Model predictions comparison
    ax7 = plt.subplot(3, 4, 7)
    models = ['Improved', 'Targeted', 'Enhanced', 'Ensemble', 'True']
    predictions = [improved_pred, targeted_pred, enhanced_pred, final_pred, true_value]
    colors = ['red', 'orange', 'green', 'blue', 'black']
    
    bars = ax7.bar(models, predictions, color=colors, alpha=0.7)
    ax7.set_title('Model Predictions Comparison')
    ax7.set_ylabel('Prediction (%)')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, predictions):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2 if value >= 0 else bar.get_height() - 15, 
                f'{value:.1f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)
    
    # 8. Error analysis
    ax8 = plt.subplot(3, 4, 8)
    errors = [abs(pred - true_value) for pred in [improved_pred, targeted_pred, enhanced_pred, final_pred]]
    model_names = ['Improved', 'Targeted', 'Enhanced', 'Ensemble']
    
    bars = ax8.bar(model_names, errors, color=['red', 'orange', 'green', 'blue'], alpha=0.7)
    ax8.set_title('Absolute Errors')
    ax8.set_ylabel('Error (%)')
    ax8.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, error in zip(bars, errors):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{error:.1f}', ha='center', va='bottom')
    
    # 9. Feature importance visualization
    ax9 = plt.subplot(3, 4, 9)
    
    # Get enhanced features
    enhanced_features = enhanced_preds[image_index]['features']
    key_features = {
        'Contour Offset': abs(enhanced_features['contour_offset_x_norm']),
        'Intensity Asymmetry': enhanced_features['intensity_asymmetry'],
        'Edge Density': enhanced_features['edge_density'],
        'Left-Right Diff': abs(enhanced_features['left_right_intensity_diff']) / 100
    }
    
    feature_names = list(key_features.keys())
    feature_values = list(key_features.values())
    
    bars = ax9.bar(feature_names, feature_values, color=['cyan', 'magenta', 'yellow', 'lime'], alpha=0.7)
    ax9.set_title('Key Features (Normalized)')
    ax9.set_ylabel('Value')
    ax9.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.setp(ax9.get_xticklabels(), rotation=45, ha='right')
    
    # 10. Prediction process flow
    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('off')
    
    process_text = f"""
PREDICTION PROCESS:

1. Feature Extraction:
   • Contour offset: {enhanced_features['contour_offset_x_norm']:.3f}
   • Intensity asymmetry: {enhanced_features['intensity_asymmetry']:.3f}
   • Edge density: {enhanced_features['edge_density']:.3f}

2. Individual Models:
   • Improved: {improved_pred:.1f}%
   • Targeted: {targeted_pred:.1f}%
   • Enhanced: {enhanced_pred:.1f}%

3. Ensemble Combination:
   • Linear regression weights
   • Advanced features included

4. Rule Application:
   • Borderline case handling
   • Final prediction: {final_pred:.1f}%

5. Result:
   • True value: {true_value:.1f}%
   • Error: {final_pred - true_value:.1f}%
    """
    
    ax10.text(0.05, 0.95, process_text, transform=ax10.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # 11. Region analysis
    ax11 = plt.subplot(3, 4, 11)
    
    # Show intensity differences
    intensity_diffs = [
        left_intensity - center_intensity,
        right_intensity - center_intensity,
        left_intensity - right_intensity
    ]
    diff_labels = ['Left-Center', 'Right-Center', 'Left-Right']
    
    bars = ax11.bar(diff_labels, intensity_diffs, color=['red', 'blue', 'green'], alpha=0.7)
    ax11.set_title('Intensity Differences')
    ax11.set_ylabel('Difference')
    ax11.grid(True, alpha=0.3)
    ax11.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, diff in zip(bars, intensity_diffs):
        ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2 if diff >= 0 else bar.get_height() - 15, 
                f'{diff:.1f}', ha='center', va='bottom' if diff >= 0 else 'top', fontsize=8)
    
    # 12. Final result
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    result_text = f"""
FINAL RESULT:

Image: {image_path}
True Value: {true_value:.1f}%
Predicted: {final_pred:.1f}%
Error: {final_pred - true_value:.1f}%
Accuracy: {100 - abs(final_pred - true_value):.1f}%

Model Performance:
• MAE: 6.04%
• RMSE: 13.82%

This prediction was made by:
1. Combining 3 base models
2. Adding advanced features
3. Applying refined rules
4. Ensemble learning
    """
    
    ax12.text(0.05, 0.95, result_text, transform=ax12.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'prediction_visualization_{image_path}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'prediction_visualization_{image_path}.png'")
    
    return {
        'true_value': true_value,
        'final_prediction': final_pred,
        'error': final_pred - true_value,
        'accuracy': 100 - abs(final_pred - true_value)
    }

def main():
    # Choose an image to visualize (you can change the index)
    image_index = 0  # 1.jpg
    image_path = "1.jpg"
    
    result = visualize_prediction_process(image_path, image_index)
    
    print(f"\n=== SUMMARY ===")
    print(f"Image: {image_path}")
    print(f"True Value: {result['true_value']:.1f}%")
    print(f"Predicted: {result['final_prediction']:.1f}%")
    print(f"Error: {result['error']:.1f}%")
    print(f"Accuracy: {result['accuracy']:.1f}%")

if __name__ == "__main__":
    main() 