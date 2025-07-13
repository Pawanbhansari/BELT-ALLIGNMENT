import json
import pandas as pd
import numpy as np

def main():
    print("=== MODEL COMPARISON SUMMARY ===\n")
    
    # Load ground truth
    df = pd.read_csv('labels.csv')
    
    # Load predictions from each model
    results = []
    
    # Load segmentation predictions
    if os.path.exists('segmentation_predictions.json'):
        with open('segmentation_predictions.json', 'r') as f:
            seg_preds = json.load(f)
        belt_seg_preds = [p for p in seg_preds if p['filename'].endswith('.jpg')]
        
        for pred in belt_seg_preds:
            filename = pred['filename']
            gt_row = df[df['filename'] == filename]
            if not gt_row.empty:
                true_val = gt_row['center_percent'].iloc[0]
                pred_val = pred['predicted_value']
                error = abs(pred_val - true_val)
                results.append({
                    'filename': filename,
                    'true': true_val,
                    'segmentation': pred_val,
                    'segmentation_error': error
                })
    
    # Load improved predictions
    if os.path.exists('improved_predictions.json'):
        with open('improved_predictions.json', 'r') as f:
            improved_preds = json.load(f)
        
        for pred in improved_preds:
            filename = pred['filename']
            if filename.endswith('.jpg'):
                # Find matching result
                for result in results:
                    if result['filename'] == filename:
                        result['improved'] = pred['predicted_value']
                        result['improved_error'] = abs(pred['predicted_value'] - result['true'])
                        break
    
    # Load targeted fix predictions
    if os.path.exists('targeted_fix_predictions.json'):
        with open('targeted_fix_predictions.json', 'r') as f:
            targeted_preds = json.load(f)
        
        for pred in targeted_preds:
            filename = pred['filename']
            # Find matching result
            for result in results:
                if result['filename'] == filename:
                    result['targeted_fix'] = pred['predicted']
                    result['targeted_fix_error'] = abs(pred['predicted'] - result['true'])
                    break
    
    # Create comparison table
    comparison_data = []
    for result in results:
        row = {
            'Filename': result['filename'],
            'True Value (%)': f"{result['true']:.1f}",
            'Segmentation (%)': f"{result.get('segmentation', 'N/A')}",
            'Improved (%)': f"{result.get('improved', 'N/A')}",
            'Targeted Fix (%)': f"{result.get('targeted_fix', 'N/A')}"
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print("DETAILED PREDICTIONS:")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS:")
    print("=" * 80)
    
    metrics = {}
    
    # Segmentation metrics
    seg_errors = [r['segmentation_error'] for r in results if 'segmentation_error' in r]
    if seg_errors:
        seg_mae = np.mean(seg_errors)
        seg_rmse = np.sqrt(np.mean([e**2 for e in seg_errors]))
        metrics['Segmentation'] = {'MAE': seg_mae, 'RMSE': seg_rmse}
        print(f"Segmentation Model:")
        print(f"  MAE: {seg_mae:.2f}%")
        print(f"  RMSE: {seg_rmse:.2f}%")
    
    # Improved metrics
    improved_errors = [r['improved_error'] for r in results if 'improved_error' in r]
    if improved_errors:
        improved_mae = np.mean(improved_errors)
        improved_rmse = np.sqrt(np.mean([e**2 for e in improved_errors]))
        metrics['Improved'] = {'MAE': improved_mae, 'RMSE': improved_rmse}
        print(f"\nImproved Model:")
        print(f"  MAE: {improved_mae:.2f}%")
        print(f"  RMSE: {improved_rmse:.2f}%")
    
    # Targeted fix metrics
    targeted_errors = [r['targeted_fix_error'] for r in results if 'targeted_fix_error' in r]
    if targeted_errors:
        targeted_mae = np.mean(targeted_errors)
        targeted_rmse = np.sqrt(np.mean([e**2 for e in targeted_errors]))
        metrics['Targeted Fix'] = {'MAE': targeted_mae, 'RMSE': targeted_rmse}
        print(f"\nTargeted Fix Model:")
        print(f"  MAE: {targeted_mae:.2f}%")
        print(f"  RMSE: {targeted_rmse:.2f}%")
    
    # Show improvements
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS:")
    print("=" * 80)
    
    if 'Segmentation' in metrics and 'Improved' in metrics:
        seg_mae = metrics['Segmentation']['MAE']
        improved_mae = metrics['Improved']['MAE']
        improvement_1 = ((seg_mae - improved_mae) / seg_mae) * 100
        print(f"Improved vs Segmentation: {improvement_1:.1f}% better")
    
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
    
    # Find best model
    if metrics:
        best_model = min(metrics.keys(), key=lambda x: metrics[x]['MAE'])
        best_mae = metrics[best_model]['MAE']
        print(f"\n🏆 BEST MODEL: {best_model} (MAE: {best_mae:.2f}%)")
    
    # Show sample-by-sample comparison
    print("\n" + "=" * 80)
    print("SAMPLE-BY-SAMPLE COMPARISON:")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result['filename']}:")
        print(f"  True Value: {result['true']:.1f}%")
        
        if 'segmentation' in result:
            print(f"  Segmentation: {result['segmentation']:.1f}% (Error: {result['segmentation_error']:.1f}%)")
        
        if 'improved' in result:
            print(f"  Improved: {result['improved']:.1f}% (Error: {result['improved_error']:.1f}%)")
        
        if 'targeted_fix' in result:
            print(f"  Targeted Fix: {result['targeted_fix']:.1f}% (Error: {result['targeted_fix_error']:.1f}%)")

if __name__ == "__main__":
    import os
    main() 