import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from torchvision import transforms

from regression_model import BeltAlignmentRegressor

class BeltAlignmentRegressionPredictor:
    def __init__(self, model_path, config_path=None):
        """
        Initialize the regression predictor with a trained model
        
        Args:
            model_path (str): Path to the trained regression model checkpoint
            config_path (str): Path to the training configuration file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                'model_name': 'resnet18',
                'img_size': 224
            }
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((self.config['img_size'], self.config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """Load the trained regression model"""
        # Create model
        model = BeltAlignmentRegressor(pretrained=False, model_name=self.config['model_name'])
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Regression model loaded from {model_path}")
        return model
    
    def predict_image(self, image_path, show_result=True):
        """
        Predict alignment percentage for a single image
        
        Args:
            image_path (str): Path to the image file
            show_result (bool): Whether to display the result
            
        Returns:
            dict: Prediction results with alignment percentage
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_value = output.item()
        
        # Interpret the result
        result = {
            'predicted_value': predicted_value,
            'alignment_status': self._interpret_alignment(predicted_value),
            'direction': self._get_direction(predicted_value),
            'severity': self._get_severity(predicted_value)
        }
        
        # Display result if requested
        if show_result:
            self._display_prediction(image_path, result, original_image)
        
        return result
    
    def _interpret_alignment(self, value):
        """Interpret alignment value"""
        if value < -50:
            return "Severe Left Misalignment"
        elif value < 0:
            return "Mild Left Misalignment"
        elif value <= 100:
            return "Good Alignment"
        else:
            return "Severe Right Misalignment"
    
    def _get_direction(self, value):
        """Get misalignment direction"""
        if value < 0:
            return "Left"
        elif value > 0:
            return "Right"
        else:
            return "Center"
    
    def _get_severity(self, value):
        """Get misalignment severity"""
        abs_value = abs(value)
        if abs_value < 10:
            return "Minimal"
        elif abs_value < 50:
            return "Mild"
        elif abs_value < 100:
            return "Moderate"
        else:
            return "Severe"
    
    def _display_prediction(self, image_path, result, original_image):
        """Display the prediction result"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        ax1.imshow(original_image)
        ax1.set_title(f'Input Image: {os.path.basename(image_path)}')
        ax1.axis('off')
        
        # Prediction results
        ax2.text(0.5, 0.7, f'Predicted Alignment: {result["predicted_value"]:.1f}%', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        
        ax2.text(0.5, 0.5, f'Status: {result["alignment_status"]}', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        ax2.text(0.5, 0.3, f'Direction: {result["direction"]}', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12)
        
        ax2.text(0.5, 0.1, f'Severity: {result["severity"]}', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12)
        
        # Add color-coded background based on alignment
        if result["predicted_value"] < -50:
            ax2.set_facecolor('lightcoral')  # Red for severe left
        elif result["predicted_value"] < 0:
            ax2.set_facecolor('lightyellow')  # Yellow for mild left
        elif result["predicted_value"] <= 100:
            ax2.set_facecolor('lightgreen')  # Green for good
        else:
            ax2.set_facecolor('lightcoral')  # Red for severe right
        
        ax2.set_title('Regression Prediction')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def predict_batch(self, image_dir, output_file=None):
        """
        Predict alignment for all images in a directory
        
        Args:
            image_dir (str): Directory containing images
            output_file (str): Optional file to save results
            
        Returns:
            list: List of prediction results
        """
        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for filename in os.listdir(image_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(image_dir, filename)
                try:
                    result = self.predict_image(image_path, show_result=False)
                    result['filename'] = filename
                    results.append(result)
                    print(f"{filename}: {result['predicted_value']:.1f}% ({result['alignment_status']})")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            print(f"Results saved to {output_file}")
        
        return results
    
    def analyze_dataset(self, csv_file, img_dir):
        """
        Analyze the entire dataset and compare predictions with ground truth
        
        Args:
            csv_file (str): Path to the CSV file with ground truth
            img_dir (str): Directory containing images
        """
        import pandas as pd
        
        # Load ground truth
        df = pd.read_csv(csv_file)
        
        predictions = []
        ground_truth = []
        
        print("Analyzing dataset with regression model...")
        for i, row in df.iterrows():
            img_path = os.path.join(img_dir, row['filename'])
            if os.path.exists(img_path):
                try:
                    result = self.predict_image(img_path, show_result=False)
                    predictions.append(result['predicted_value'])
                    ground_truth.append(row['center_percent'])
                except Exception as e:
                    print(f"Error processing {row['filename']}: {e}")
        
        if predictions:
            # Calculate metrics
            predictions = np.array(predictions)
            ground_truth = np.array(ground_truth)
            
            mae = np.mean(np.abs(predictions - ground_truth))
            rmse = np.sqrt(np.mean((predictions - ground_truth)**2))
            
            # Calculate R-squared
            ss_res = np.sum((ground_truth - predictions) ** 2)
            ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            print(f"\nDataset Analysis Results:")
            print(f"Total samples: {len(predictions)}")
            print(f"Mean Absolute Error (MAE): {mae:.2f}")
            print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
            print(f"R-squared (R²): {r2:.4f}")
            
            # Plot results
            plt.figure(figsize=(12, 5))
            
            # Scatter plot
            plt.subplot(1, 2, 1)
            plt.scatter(ground_truth, predictions, alpha=0.7, color='blue', s=100)
            plt.plot([ground_truth.min(), ground_truth.max()], 
                    [ground_truth.min(), ground_truth.max()], 'r--', lw=2)
            plt.xlabel('True Alignment (%)')
            plt.ylabel('Predicted Alignment (%)')
            plt.title('Regression Predictions vs True Values')
            plt.grid(True, alpha=0.3)
            
            # Error distribution
            plt.subplot(1, 2, 2)
            errors = predictions - ground_truth
            plt.hist(errors, bins=10, alpha=0.7, color='orange', edgecolor='black')
            plt.xlabel('Prediction Error (%)')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('regression_dataset_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print detailed comparison
            print(f"\nDetailed Predictions:")
            for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
                error = pred - true
                print(f"Sample {i+1}: True={true:.1f}%, Predicted={pred:.1f}%, Error={error:.1f}%")
            
            return predictions, ground_truth, mae, rmse, r2
        
        return None, None, None, None, None

def main():
    # Example usage
    model_path = 'best_regression_model.pth'  # or 'final_regression_model.pth'
    config_path = 'regression_config.json'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the regression model first.")
        print("Run: python regression_train.py")
        return
    
    # Initialize predictor
    predictor = BeltAlignmentRegressionPredictor(model_path, config_path)
    
    # Example 1: Predict single image
    if os.path.exists('1.jpg'):
        print("\nPredicting single image...")
        result = predictor.predict_image('1.jpg')
        print(f"Result: {result}")
    
    # Example 2: Predict all images in current directory
    print("\nPredicting all images in current directory...")
    results = predictor.predict_batch('.', 'regression_predictions.json')
    
    # Example 3: Analyze dataset if labels.csv exists
    if os.path.exists('labels.csv'):
        print("\nAnalyzing dataset...")
        predictor.analyze_dataset('labels.csv', '.')

if __name__ == "__main__":
    main() 