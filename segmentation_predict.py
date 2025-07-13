import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from torchvision import transforms

from segmentation_model import BeltSegmentationUNet

class BeltSegmentationPredictor:
    def __init__(self, model_path, config_path=None):
        """
        Initialize the segmentation predictor with a trained model
        
        Args:
            model_path (str): Path to the trained segmentation model checkpoint
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
        """Load the trained segmentation model"""
        # Create model
        model = BeltSegmentationUNet(num_classes=2, pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Segmentation model loaded from {model_path}")
        return model
    
    def predict_image(self, image_path, show_result=True):
        """
        Predict belt segmentation and calculate misalignment
        
        Args:
            image_path (str): Path to the image file
            show_result (bool): Whether to display the result
            
        Returns:
            dict: Prediction results with segmentation and misalignment analysis
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        original_size = image.size
        
        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get segmentation prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Resize mask to original image size
        pred_mask_resized = cv2.resize(pred_mask, original_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        # Analyze segmentation to calculate misalignment
        misalignment_analysis = self.analyze_belt_misalignment(pred_mask_resized, original_size)
        
        # Display result if requested
        if show_result:
            self._display_prediction(image_path, original_image, pred_mask_resized, misalignment_analysis)
        
        return misalignment_analysis
    
    def analyze_belt_misalignment(self, mask, img_size):
        """
        Analyze belt segmentation mask to calculate misalignment percentage
        
        Args:
            mask: Binary segmentation mask (0=background, 1=belt)
            img_size: Original image size (width, height)
            
        Returns:
            dict: Misalignment analysis results
        """
        width, height = img_size
        
        # Find belt pixels
        belt_pixels = np.where(mask == 1)
        
        if len(belt_pixels[0]) == 0:
            return {
                'predicted_value': 0.0,
                'alignment_status': 'No Belt Detected',
                'direction': 'Unknown',
                'severity': 'Unknown',
                'belt_center': width // 2,
                'image_center': width // 2,
                'offset_pixels': 0,
                'belt_area': 0
            }
        
        # Calculate belt center
        belt_center_x = np.mean(belt_pixels[1])  # Column (x) coordinates
        belt_center_y = np.mean(belt_pixels[0])  # Row (y) coordinates
        
        # Calculate image center
        image_center_x = width // 2
        image_center_y = height // 2
        
        # Calculate offset
        offset_pixels = belt_center_x - image_center_x
        
        # Convert offset to percentage
        # Normalize by image width and scale to match your data range
        max_offset = width * 0.4  # 40% of image width as maximum
        predicted_percentage = (offset_pixels / max_offset) * 100
        
        # Clamp to reasonable range
        predicted_percentage = np.clip(predicted_percentage, -200, 200)
        
        # Determine alignment status
        if predicted_percentage < -50:
            alignment_status = "Severe Left Misalignment"
            direction = "Left"
            severity = "Severe"
        elif predicted_percentage < 0:
            alignment_status = "Mild Left Misalignment"
            direction = "Left"
            severity = "Mild"
        elif predicted_percentage <= 100:
            alignment_status = "Good Alignment"
            direction = "Center"
            severity = "Good"
        else:
            alignment_status = "Severe Right Misalignment"
            direction = "Right"
            severity = "Severe"
        
        return {
            'predicted_value': float(predicted_percentage),
            'alignment_status': alignment_status,
            'direction': direction,
            'severity': severity,
            'belt_center': float(belt_center_x),
            'image_center': float(image_center_x),
            'offset_pixels': float(offset_pixels),
            'belt_area': len(belt_pixels[0])
        }
    
    def _display_prediction(self, image_path, original_image, mask, analysis):
        """Display the prediction result with segmentation and analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f'Original Image: {os.path.basename(image_path)}')
        axes[0, 0].axis('off')
        
        # Segmentation mask
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Belt Segmentation Mask')
        axes[0, 1].axis('off')
        
        # Overlay mask on original image
        overlay = original_image.copy()
        # Ensure mask is in the correct format for cv2.resize
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        # Resize mask to match original image dimensions
        mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Ensure mask is binary
        mask_resized = (mask_resized > 0.5).astype(np.uint8)
        overlay[mask_resized == 1] = [255, 0, 0]  # Red for belt pixels
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Segmentation Overlay')
        axes[1, 0].axis('off')
        
        # Analysis results
        ax = axes[1, 1]
        ax.text(0.5, 0.9, f'Predicted Alignment: {analysis["predicted_value"]:.1f}%', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        
        ax.text(0.5, 0.7, f'Status: {analysis["alignment_status"]}', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        ax.text(0.5, 0.5, f'Direction: {analysis["direction"]}', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12)
        
        ax.text(0.5, 0.3, f'Severity: {analysis["severity"]}', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12)
        
        ax.text(0.5, 0.1, f'Belt Area: {analysis["belt_area"]} pixels', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10)
        
        # Add color-coded background based on alignment
        if analysis["predicted_value"] < -50:
            ax.set_facecolor('lightcoral')  # Red for severe left
        elif analysis["predicted_value"] < 0:
            ax.set_facecolor('lightyellow')  # Yellow for mild left
        elif analysis["predicted_value"] <= 100:
            ax.set_facecolor('lightgreen')  # Green for good
        else:
            ax.set_facecolor('lightcoral')  # Red for severe right
        
        ax.set_title('Segmentation Analysis')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def predict_batch(self, image_dir, output_file=None):
        """
        Predict alignment for all images in a directory using segmentation
        
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
        
        print("Analyzing dataset with segmentation model...")
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
            
            print(f"\nSegmentation Dataset Analysis Results:")
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
            plt.title('Segmentation Predictions vs True Values')
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
            plt.savefig('segmentation_dataset_analysis.png', dpi=300, bbox_inches='tight')
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
    model_path = 'best_segmentation_model.pth'  # or 'final_segmentation_model.pth'
    config_path = 'segmentation_config.json'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the segmentation model first.")
        print("Run: python segmentation_train.py")
        return
    
    # Initialize predictor
    predictor = BeltSegmentationPredictor(model_path, config_path)
    
    # Example 1: Predict single image
    if os.path.exists('1.jpg'):
        print("\nPredicting single image with segmentation...")
        result = predictor.predict_image('1.jpg')
        print(f"Result: {result}")
    
    # Example 2: Predict all images in current directory
    print("\nPredicting all images in current directory...")
    results = predictor.predict_batch('.', 'segmentation_predictions.json')
    
    # Example 3: Analyze dataset if labels.csv exists
    if os.path.exists('labels.csv'):
        print("\nAnalyzing dataset...")
        predictor.analyze_dataset('labels.csv', '.')

if __name__ == "__main__":
    main() 