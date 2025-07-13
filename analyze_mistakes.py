#!/usr/bin/env python3
"""
Analyze model mistakes and provide detailed insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

def analyze_mistakes():
    """Analyze which images the model is making mistakes on"""
    
    # Load ground truth
    df = pd.read_csv('labels.csv')
    
    # Load predictions (only for actual images, not PNG files)
    with open('predictions.json', 'r') as f:
        predictions = json.load(f)
    
    # Filter predictions to only include actual images
    image_predictions = [p for p in predictions if p['filename'].endswith('.jpg')]
    
    print("=" * 60)
    print("MISTAKE ANALYSIS")
    print("=" * 60)
    
    # Class mapping
    class_names = ['Severe Left Misalignment', 'Mild Left Misalignment', 'Good Alignment', 'Severe Right Misalignment']
    
    # Analyze each image
    mistakes = []
    correct = []
    
    for i, row in df.iterrows():
        filename = row['filename']
        true_value = row['center_percent']
        
        # Find corresponding prediction
        pred = None
        for p in image_predictions:
            if p['filename'] == filename:
                pred = p
                break
        
        if pred is None:
            continue
        
        # Determine true class
        if true_value < -50:
            true_class = 0  # Severe Left
        elif true_value < 0:
            true_class = 1  # Mild Left
        elif true_value <= 100:
            true_class = 2  # Good
        else:
            true_class = 3  # Severe Right
        
        predicted_class = pred['predicted_class']
        confidence = pred['confidence']
        
        # Check if prediction is correct
        is_correct = (true_class == predicted_class)
        
        result = {
            'filename': filename,
            'true_value': true_value,
            'true_class': true_class,
            'true_class_name': class_names[true_class],
            'predicted_class': predicted_class,
            'predicted_class_name': class_names[predicted_class],
            'confidence': confidence,
            'is_correct': is_correct,
            'probabilities': pred['all_probabilities']
        }
        
        if is_correct:
            correct.append(result)
        else:
            mistakes.append(result)
    
    # Print results
    print(f"\nTotal images: {len(df)}")
    print(f"Correct predictions: {len(correct)}")
    print(f"Mistakes: {len(mistakes)}")
    print(f"Accuracy: {len(correct)/(len(correct)+len(mistakes))*100:.1f}%")
    
    if mistakes:
        print(f"\n{'='*60}")
        print("MISTAKES DETAILED ANALYSIS")
        print(f"{'='*60}")
        
        for mistake in mistakes:
            print(f"\n📸 {mistake['filename']}")
            print(f"   True: {mistake['true_class_name']} (value: {mistake['true_value']})")
            print(f"   Predicted: {mistake['predicted_class_name']} (confidence: {mistake['confidence']:.1%})")
            print(f"   Error: {mistake['true_class_name']} → {mistake['predicted_class_name']}")
            
            # Analyze probabilities
            if isinstance(mistake['probabilities'], str):
                # Convert string representation to list
                prob_str = mistake['probabilities'].replace('[', '').replace(']', '')
                probs = [float(x) for x in prob_str.split()]
            else:
                probs = mistake['probabilities']
            print(f"   Probabilities:")
            for i, prob in enumerate(probs):
                marker = "✓" if i == mistake['predicted_class'] else "  "
                print(f"     {marker} {class_names[i]}: {prob:.1%}")
    
    if correct:
        print(f"\n{'='*60}")
        print("CORRECT PREDICTIONS")
        print(f"{'='*60}")
        
        for correct_pred in correct:
            print(f"✓ {correct_pred['filename']}: {correct_pred['true_class_name']} (confidence: {correct_pred['confidence']:.1%})")
    
    # Visualize mistakes
    if mistakes:
        visualize_mistakes(mistakes, correct)
    
    return mistakes, correct

def visualize_mistakes(mistakes, correct):
    """Create visualizations of the mistakes"""
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Accuracy breakdown
    ax1 = axes[0, 0]
    labels = ['Correct', 'Mistakes']
    sizes = [len(correct), len(mistakes)]
    colors = ['lightgreen', 'lightcoral']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Prediction Accuracy')
    
    # 2. Confidence distribution
    ax2 = axes[0, 1]
    correct_confidences = [c['confidence'] for c in correct]
    mistake_confidences = [m['confidence'] for m in mistakes]
    
    if correct_confidences:
        ax2.hist(correct_confidences, alpha=0.7, label='Correct', color='green', bins=10)
    if mistake_confidences:
        ax2.hist(mistake_confidences, alpha=0.7, label='Mistakes', color='red', bins=10)
    
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error type analysis
    ax3 = axes[1, 0]
    if mistakes:
        error_types = {}
        for mistake in mistakes:
            error_key = f"{mistake['true_class_name']} → {mistake['predicted_class_name']}"
            error_types[error_key] = error_types.get(error_key, 0) + 1
        
        error_names = list(error_types.keys())
        error_counts = list(error_types.values())
        
        bars = ax3.bar(range(len(error_names)), error_counts, color='lightcoral')
        ax3.set_xlabel('Error Type')
        ax3.set_ylabel('Count')
        ax3.set_title('Types of Mistakes')
        ax3.set_xticks(range(len(error_names)))
        ax3.set_xticklabels(error_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, error_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(count), ha='center', va='bottom')
    
    # 4. Value vs prediction scatter
    ax4 = axes[1, 1]
    
    # Combine all results
    all_results = correct + mistakes
    true_values = [r['true_value'] for r in all_results]
    predicted_classes = [r['predicted_class'] for r in all_results]
    colors = ['green' if r['is_correct'] else 'red' for r in all_results]
    
    ax4.scatter(true_values, predicted_classes, c=colors, alpha=0.7, s=100)
    ax4.set_xlabel('True Center Percent')
    ax4.set_ylabel('Predicted Class')
    ax4.set_title('True Values vs Predictions')
    ax4.grid(True, alpha=0.3)
    
    # Add class boundaries
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Severe Left boundary')
    ax4.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='Mild Left boundary')
    ax4.axhline(y=2.5, color='green', linestyle='--', alpha=0.5, label='Good boundary')
    ax4.axhline(y=3.5, color='red', linestyle='--', alpha=0.5, label='Severe Right boundary')
    
    # Add value boundaries
    ax4.axvline(x=-50, color='red', linestyle=':', alpha=0.5)
    ax4.axvline(x=0, color='orange', linestyle=':', alpha=0.5)
    ax4.axvline(x=100, color='green', linestyle=':', alpha=0.5)
    
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('mistake_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def suggest_improvements(mistakes):
    """Suggest improvements based on the mistakes"""
    
    print(f"\n{'='*60}")
    print("SUGGESTIONS FOR IMPROVEMENT")
    print(f"{'='*60}")
    
    if not mistakes:
        print("🎉 No mistakes found! The model is performing well.")
        return
    
    # Analyze mistake patterns
    error_patterns = {}
    for mistake in mistakes:
        error_type = f"{mistake['true_class_name']} → {mistake['predicted_class_name']}"
        error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
    
    print(f"\nMost common mistakes:")
    for error_type, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {error_type}: {count} times")
    
    # Specific suggestions
    print(f"\nSpecific suggestions:")
    
    # Check for boundary confusion
    boundary_mistakes = []
    for mistake in mistakes:
        true_val = mistake['true_value']
        if abs(true_val) < 20:  # Close to boundaries
            boundary_mistakes.append(mistake)
    
    if boundary_mistakes:
        print(f"  • {len(boundary_mistakes)} mistakes near class boundaries")
        print(f"    → Consider adjusting classification thresholds")
        print(f"    → Add more training data near boundaries")
    
    # Check for low confidence mistakes
    low_conf_mistakes = [m for m in mistakes if m['confidence'] < 0.6]
    if low_conf_mistakes:
        print(f"  • {len(low_conf_mistakes)} mistakes with low confidence (< 60%)")
        print(f"    → Model is uncertain about these cases")
        print(f"    → Consider ensemble methods or additional features")
    
    # General suggestions
    print(f"\nGeneral improvements:")
    print(f"  • Increase dataset size (currently only {len(mistakes) + len(correct)} samples)")
    print(f"  • Add more data augmentation")
    print(f"  • Try different model architectures")
    print(f"  • Use cross-validation for better evaluation")
    print(f"  • Consider regression instead of classification for fine-grained predictions")

if __name__ == "__main__":
    mistakes, correct = analyze_mistakes()
    suggest_improvements(mistakes) 