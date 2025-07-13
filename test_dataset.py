#!/usr/bin/env python3
"""
Test script to verify dataset loading and visualization
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.append('.')

from data_loader import BeltAlignmentDataset, get_transforms, visualize_dataset

def test_dataset_loading():
    """Test basic dataset loading functionality"""
    print("Testing dataset loading...")
    
    # Check if required files exist
    if not os.path.exists('labels.csv'):
        print("ERROR: labels.csv not found!")
        return False
    
    # Count images
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} image files: {image_files}")
    
    # Test dataset creation
    try:
        dataset = BeltAlignmentDataset('labels.csv', '.', transform=None, classification_mode=True)
        print(f"✓ Dataset created successfully with {len(dataset)} samples")
        
        # Test class distribution
        class_counts = {}
        for i in range(len(dataset)):
            _, label = dataset[i]
            class_counts[label] = class_counts.get(label, 0) + 1
        
        print("Class distribution:")
        class_names = dataset.get_class_names()
        for i, count in sorted(class_counts.items()):
            print(f"  {class_names[i]}: {count} samples")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to create dataset: {e}")
        return False

def test_transforms():
    """Test image transforms"""
    print("\nTesting transforms...")
    
    try:
        train_transform, val_transform = get_transforms(img_size=224)
        print("✓ Transforms created successfully")
        
        # Test with a sample image
        dataset = BeltAlignmentDataset('labels.csv', '.', transform=val_transform, classification_mode=True)
        img, label = dataset[0]
        
        print(f"✓ Transform applied successfully")
        print(f"  Input image shape: {img.shape}")
        print(f"  Label: {label} ({dataset.get_class_names()[label]})")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to test transforms: {e}")
        return False

def test_visualization():
    """Test dataset visualization"""
    print("\nTesting visualization...")
    
    try:
        # Create dataset with transforms
        _, val_transform = get_transforms(img_size=224)
        dataset = BeltAlignmentDataset('labels.csv', '.', transform=val_transform, classification_mode=True)
        
        # Visualize dataset
        print("Displaying dataset samples...")
        visualize_dataset(dataset, num_samples=min(4, len(dataset)))
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to visualize dataset: {e}")
        return False

def analyze_labels():
    """Analyze the label distribution"""
    print("\nAnalyzing labels...")
    
    try:
        import pandas as pd
        
        # Load labels
        df = pd.read_csv('labels.csv')
        print(f"✓ Loaded {len(df)} labels from CSV")
        
        # Analyze center_percent values
        center_values = df['center_percent'].values
        print(f"Center percent statistics:")
        print(f"  Min: {center_values.min():.2f}")
        print(f"  Max: {center_values.max():.2f}")
        print(f"  Mean: {center_values.mean():.2f}")
        print(f"  Std: {center_values.std():.2f}")
        
        # Show value distribution
        print(f"\nValue distribution:")
        print(f"  < -50 (Severe Left): {sum(center_values < -50)}")
        print(f"  -50 to 0 (Mild Left): {sum((center_values >= -50) & (center_values < 0))}")
        print(f"  0 to 100 (Good): {sum((center_values >= 0) & (center_values <= 100))}")
        print(f"  > 100 (Severe Right): {sum(center_values > 100)}")
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(center_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=-50, color='red', linestyle='--', label='Severe Left threshold')
        plt.axvline(x=0, color='orange', linestyle='--', label='Mild Left threshold')
        plt.axvline(x=100, color='green', linestyle='--', label='Good alignment threshold')
        plt.xlabel('Center Percent')
        plt.ylabel('Frequency')
        plt.title('Distribution of Belt Alignment Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to analyze labels: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("BELT ALIGNMENT DATASET TEST")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Transforms", test_transforms),
        ("Label Analysis", analyze_labels),
        ("Visualization", test_visualization)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"ERROR: {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The dataset is ready for training.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train the model: python train.py")
        print("3. Make predictions: python predict.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    main() 