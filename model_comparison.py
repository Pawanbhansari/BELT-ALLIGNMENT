import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_predictions():
    """Load predictions from all models"""
    df = pd.read_csv('labels.csv')
    
    # Load all prediction files
    predictions = {}
    
    # Segmentation model predictions
    if os.path.exists('segmentation_predictions.json'):
        with open('segmentation_predictions.json', 'r') as f:
            seg_preds = json.load(f)
        # Filter to only belt images
        belt_seg_preds = [p for p in seg_preds if p['filename'].endswith('.jpg')]
        predictions['Segmentation'] = belt_seg_preds
    
    # Improved model predictions
    if os.path.exists('improved_predictions.json'):
        with open('improved_predictions.json', 'r') as f:
            improved_preds = json.load(f)
        # Filter to only belt images
        belt_improved_preds = [p for p in improved_preds if p['filename'].endswith('.jpg')]
        predictions['Improved'] = belt_improved_preds
    
    # Targeted fix model predictions
    if os.path.exists('targeted_fix_predictions.json'):
        with open('targeted_fix_predictions.json', 'r') as f:
            targeted_preds = json.load(f)
        predictions['Targeted Fix'] = targeted_preds
    
    return df, predictions

def create_comparison_table():
    """Create a comprehensive comparison table"""
    df, predictions = load_predictions()
    
    # Create comparison dataframe
    comparison_data = []
    
    for _, row in df.iterrows():
        filename = row['filename']
        true_value = row['center_percent']
        
        row_data = {
            'Filename': filename,
            'True Value (%)': true_value
        }
        
        # Add predictions from each model
        for model_name, model_preds in predictions.items():
            pred = next((p for p in model_preds if p['filename'] == filename), None)
            if pred:
                row_data[f'{model_name} Prediction (%)'] = pred['predicted_value']
                row_data[f'{model_name} Error (%)'] = pred['predicted_value'] - true_value
                row_data[f'{model_name} Abs Error (%)'] = abs(pred['predicted_value'] - true_value)
            else:
                row_data[f'{model_name} Prediction (%)'] = None
                row_data[f'{model_name} Error (%)'] = None
                row_data[f'{model_name} Abs Error (%)'] = None
        
        comparison_data.append(row_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df

def calculate_model_metrics(comparison_df):
    """Calculate performance metrics for each model"""
    metrics = {}
    
    for model_name in ['Segmentation', 'Improved', 'Targeted Fix']:
        pred_col = f'{model_name} Prediction (%)'
        error_col = f'{model_name} Error (%)'
        abs_error_col = f'{model_name} Abs Error (%)'
        
        if pred_col in comparison_df.columns:
            # Remove rows with None values
            valid_data = comparison_df.dropna(subset=[pred_col])
            
            if len(valid_data) > 0:
                mae = valid_data[abs_error_col].mean()
                rmse = np.sqrt(np.mean(valid_data[error_col] ** 2))
                
                # Calculate R-squared
                true_values = valid_data['True Value (%)']
                pred_values = valid_data[pred_col]
                ss_res = np.sum((true_values - pred_values) ** 2)
                ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                metrics[model_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R²': r2,
                    'Samples': len(valid_data)
                }
    
    return metrics

def main():
    print("=== COMPREHENSIVE MODEL COMPARISON ===\n")
    
    # Create comparison table
    comparison_df = create_comparison_table()
    
    # Display comparison table
    print("Detailed Predictions Comparison:")
    print("=" * 80)
    
    # Format the table nicely
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 15)
    
    # Show only key columns for readability
    display_cols = ['Filename', 'True Value (%)']
    for model in ['Segmentation', 'Improved', 'Targeted Fix']:
        if f'{model} Prediction (%)' in comparison_df.columns:
            display_cols.extend([f'{model} Prediction (%)', f'{model} Error (%)'])
    
    print(comparison_df[display_cols].to_string(index=False))
    
    # Calculate and display metrics
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE METRICS:")
    print("=" * 80)
    
    metrics = calculate_model_metrics(comparison_df)
    
    # Create metrics table
    metrics_data = []
    for model_name, metric_values in metrics.items():
        metrics_data.append({
            'Model': model_name,
            'MAE (%)': f"{metric_values['MAE']:.2f}",
            'RMSE (%)': f"{metric_values['RMSE']:.2f}",
            'R²': f"{metric_values['R²']:.4f}",
            'Samples': metric_values['Samples']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df.to_string(index=False))
    
    # Find best model
    if metrics:
        best_model = min(metrics.keys(), key=lambda x: metrics[x]['MAE'])
        best_mae = metrics[best_model]['MAE']
        print(f"\n🏆 BEST MODEL: {best_model} (MAE: {best_mae:.2f}%)")
    
    # Show improvement percentages
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS:")
    print("=" * 80)
    
    if 'Segmentation' in metrics and 'Improved' in metrics:
        seg_mae = metrics['Segmentation']['MAE']
        improved_mae = metrics['Improved']['MAE']
        improvement_1 = ((seg_mae - improved_mae) / seg_mae) * 100
        print(f"Improved Model vs Segmentation: {improvement_1:.1f}% better")
    
    if 'Improved' in metrics and 'Targeted Fix' in metrics:
        improved_mae = metrics['Improved']['MAE']
        targeted_mae = metrics['Targeted Fix']['MAE']
        improvement_2 = ((improved_mae - targeted_mae) / improved_mae) * 100
        print(f"Targeted Fix vs Improved: {improvement_2:.1f}% better")
    
    if 'Segmentation' in metrics and 'Targeted Fix' in metrics:
        seg_mae = metrics['Segmentation']['MAE']
        targeted_mae = metrics['Targeted Fix']['MAE']
        total_improvement = ((seg_mae - targeted_mae) / seg_mae) * 100
        print(f"Targeted Fix vs Segmentation: {total_improvement:.1f}% better")
    
    # Create visualization
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION...")
    print("=" * 80)
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Predictions vs True Values
    plt.subplot(2, 3, 1)
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    
    for i, (model_name, color, marker) in enumerate(zip(['Segmentation', 'Improved', 'Targeted Fix'], colors, markers)):
        if f'{model_name} Prediction (%)' in comparison_df.columns:
            valid_data = comparison_df.dropna(subset=[f'{model_name} Prediction (%)'])
            if len(valid_data) > 0:
                plt.scatter(valid_data['True Value (%)'], 
                          valid_data[f'{model_name} Prediction (%)'],
                          c=color, marker=marker, s=100, alpha=0.7, label=model_name)
    
    # Add perfect prediction line
    min_val = comparison_df['True Value (%)'].min()
    max_val = comparison_df['True Value (%)'].max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('True Alignment (%)')
    plt.ylabel('Predicted Alignment (%)')
    plt.title('Predictions vs True Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Error Distribution
    plt.subplot(2, 3, 2)
    for i, (model_name, color) in enumerate(zip(['Segmentation', 'Improved', 'Targeted Fix'], colors)):
        if f'{model_name} Error (%)' in comparison_df.columns:
            valid_data = comparison_df.dropna(subset=[f'{model_name} Error (%)'])
            if len(valid_data) > 0:
                plt.hist(valid_data[f'{model_name} Error (%)'], bins=8, alpha=0.6, 
                        color=color, label=model_name, edgecolor='black')
    
    plt.xlabel('Prediction Error (%)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: MAE Comparison
    plt.subplot(2, 3, 3)
    if metrics:
        models = list(metrics.keys())
        maes = [metrics[model]['MAE'] for model in models]
        colors_mae = ['red', 'blue', 'green'][:len(models)]
        
        bars = plt.bar(models, maes, color=colors_mae, alpha=0.7, edgecolor='black')
        plt.ylabel('Mean Absolute Error (%)')
        plt.title('MAE Comparison')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mae in zip(bars, maes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{mae:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 4: Sample-wise Error Comparison
    plt.subplot(2, 3, 4)
    x_pos = range(len(comparison_df))
    width = 0.25
    
    for i, (model_name, color) in enumerate(zip(['Segmentation', 'Improved', 'Targeted Fix'], colors)):
        if f'{model_name} Abs Error (%)' in comparison_df.columns:
            valid_data = comparison_df.dropna(subset=[f'{model_name} Abs Error (%)'])
            if len(valid_data) > 0:
                plt.bar([x + i*width for x in range(len(valid_data))], 
                       valid_data[f'{model_name} Abs Error (%)'],
                       width, label=model_name, alpha=0.7, color=color)
    
    plt.xlabel('Sample Index')
    plt.ylabel('Absolute Error (%)')
    plt.title('Sample-wise Error Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: R² Comparison
    plt.subplot(2, 3, 5)
    if metrics:
        r2_values = [metrics[model]['R²'] for model in models]
        
        bars = plt.bar(models, r2_values, color=colors_mae, alpha=0.7, edgecolor='black')
        plt.ylabel('R-squared (R²)')
        plt.title('R² Comparison')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, r2 in zip(bars, r2_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 6: Improvement Summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Create text summary
    summary_text = "MODEL COMPARISON SUMMARY\n\n"
    
    if 'Segmentation' in metrics and 'Targeted Fix' in metrics:
        seg_mae = metrics['Segmentation']['MAE']
        targeted_mae = metrics['Targeted Fix']['MAE']
        total_improvement = ((seg_mae - targeted_mae) / seg_mae) * 100
        
        summary_text += f"Segmentation Model:\n"
        summary_text += f"  MAE: {seg_mae:.2f}%\n\n"
        summary_text += f"Targeted Fix Model:\n"
        summary_text += f"  MAE: {targeted_mae:.2f}%\n\n"
        summary_text += f"Total Improvement:\n"
        summary_text += f"  {total_improvement:.1f}% better"
    
    plt.text(0.5, 0.5, summary_text, ha='center', va='center', 
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed comparison to CSV
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print(f"\nDetailed comparison saved to 'model_comparison_results.csv'")
    print(f"Visualization saved to 'model_comparison.png'")
    
    return comparison_df, metrics

if __name__ == "__main__":
    comparison_df, metrics = main() 