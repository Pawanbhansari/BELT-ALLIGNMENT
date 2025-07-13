import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from torchvision import transforms

from model import get_model
from data_loader import BeltAlignmentDataset

class BeltAlignmentPredictor:
    def __init__(self, model_path, config_path=None):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path (str): Path to the trained model checkpoint
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
                'task': 'classification',
                'model_type': 'classifier',
                'model_name': 'resnet18',
                'num_classes': 4,
                'img_size': 224
            }
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Class names
        self.class_names = ['Severe Left Misalignment', 'Mild Left Misalignment', 
                           'Good Alignment', 'Severe Right Misalignment']
        
        # Transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((self.config['img_size'], self.config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """Load the trained model"""
        # Create model
        model = get_model(
            model_type=self.config['model_type'],
            num_classes=self.config['num_classes'],
            pretrained=False,
            model_name=self.config['model_name']
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Model loaded from {model_path}")
        return model
    
    def predict_image(self, image_path, show_result=True):
        """
        Predict alignment class for a single image
        
        Args:
            image_path (str): Path to the image file
            show_result (bool): Whether to display the result
            
        Returns:
            dict: Prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            
            if self.config['task'] == 'classification':
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                result = {
                    'predicted_class': predicted_class,
                    'class_name': self.class_names[predicted_class],
                    'confidence': confidence,
                    'all_probabilities': probabilities[0].cpu().numpy()
                }
            else:
                # Regression task
                predicted_value = output.item()
                result = {
                    'predicted_value': predicted_value,
                    'alignment_status': self._interpret_regression_value(predicted_value)
                }
        
        # Display result if requested
        if show_result:
            self._display_prediction(image_path, result, original_image)
        
        return result
    
    def _interpret_regression_value(self, value):
        """Interpret regression value for alignment status"""
        if value < -50:
            return "Severe Left Misalignment"
        elif value < 0:
            return "Mild Left Misalignment"
        elif value <= 100:
            return "Good Alignment"
        else:
            return "Severe Right Misalignment"
    
    def _display_prediction(self, image_path, result, original_image):
        """Display the prediction result"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        ax1.imshow(original_image)
        ax1.set_title(f'Input Image: {os.path.basename(image_path)}')
        ax1.axis('off')
        
        # Prediction results
        if self.config['task'] == 'classification':
            # Bar chart of probabilities
            classes = [name.split()[0] for name in self.class_names]  # Short names
            probabilities = result['all_probabilities']
            
            bars = ax2.bar(classes, probabilities, color=['red', 'orange', 'green', 'red'])
            ax2.set_title(f'Prediction: {result["class_name"]}\nConfidence: {result["confidence"]:.2%}')
            ax2.set_ylabel('Probability')
            ax2.set_ylim(0, 1)
            
            # Color bars based on prediction
            for i, bar in enumerate(bars):
                if i == result['predicted_class']:
                    bar.set_color('green')
                elif probabilities[i] > 0.1:  # Highlight significant probabilities
                    bar.set_color('orange')
                else:
                    bar.set_color('lightgray')
            
            # Add probability values on bars
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.2%}', ha='center', va='bottom')
        else:
            # Regression result
            ax2.text(0.5, 0.5, f'Predicted Value: {result["predicted_value"]:.2f}\n'
                               f'Status: {result["alignment_status"]}', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
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
                    print(f"{filename}: {result.get('class_name', result.get('alignment_status', 'Unknown'))}")
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
        # Load dataset
        dataset = BeltAlignmentDataset(csv_file, img_dir, transform=self.transform, 
                                     classification_mode=True)
        
        predictions = []
        ground_truth = []
        
        print("Analyzing dataset...")
        for i in range(len(dataset)):
            img, label = dataset[i]
            img = img.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img)
                if self.config['task'] == 'classification':
                    pred = torch.argmax(output, dim=1).item()
                    predictions.append(pred)
                    ground_truth.append(label)
        
        # Calculate accuracy
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        accuracy = correct / len(predictions)
        
        print(f"\nDataset Analysis Results:")
        print(f"Total samples: {len(predictions)}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        
        # Show confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(ground_truth, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix - Dataset Analysis')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return predictions, ground_truth, accuracy

def main():
    # Example usage
    model_path = 'best_model.pth'  # or 'final_model.pth'
    config_path = 'training_config.json'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    # Initialize predictor
    predictor = BeltAlignmentPredictor(model_path, config_path)
    
    # Example 1: Predict single image
    if os.path.exists('1.jpg'):
        print("\nPredicting single image...")
        result = predictor.predict_image('1.jpg')
        print(f"Result: {result}")
    
    # Example 2: Predict all images in current directory
    print("\nPredicting all images in current directory...")
    results = predictor.predict_batch('.', 'predictions.json')
    
    # Example 3: Analyze dataset if labels.csv exists
    if os.path.exists('labels.csv'):
        print("\nAnalyzing dataset...")
        predictor.analyze_dataset('labels.csv', '.')

if __name__ == "__main__":
    main() 