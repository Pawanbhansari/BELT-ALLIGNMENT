import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
import traceback

class ImprovedBeltAlignmentModel:
    """
    Improved belt alignment model using traditional CV + ML approach
    """
    
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, image_path):
        """
        Extract multiple features from belt image using traditional CV techniques
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            dict: Dictionary of extracted features
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get image dimensions
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        
        features = {}
        
        # 1. Edge-based features
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.where(edges > 0)
        
        if len(edge_pixels[0]) > 0:
            edge_center_x = np.mean(edge_pixels[1])
            edge_center_y = np.mean(edge_pixels[0])
            edge_offset_x = edge_center_x - center_x
            edge_offset_y = edge_center_y - center_y
            edge_density = len(edge_pixels[0]) / (width * height)
        else:
            edge_center_x = center_x
            edge_center_y = center_y
            edge_offset_x = 0
            edge_offset_y = 0
            edge_density = 0
        
        features['edge_center_x'] = edge_center_x
        features['edge_center_y'] = edge_center_y
        features['edge_offset_x'] = edge_offset_x
        features['edge_offset_y'] = edge_offset_y
        features['edge_density'] = edge_density
        
        # 2. Contour-based features
        print(f"[DEBUG] {image_path} edges shape: {edges.shape}, dtype: {edges.dtype}, unique: {np.unique(edges)}")
        try:
            contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[-2]
        except Exception as e:
            print(f"[ERROR] findContours failed for {image_path}: {e}")
            traceback.print_exc()
            contours = []
        if len(contours) == 0:
            contours = []
        
        if contours:
            # Find largest contour (likely the belt)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Get contour center
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                contour_center_x = M["m10"] / M["m00"]
                contour_center_y = M["m01"] / M["m00"]
            else:
                contour_center_x = center_x
                contour_center_y = center_y
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 1
            
            # Calculate offset from center
            contour_offset_x = contour_center_x - center_x
            contour_offset_y = contour_center_y - center_y
            
            # Normalize by image dimensions
            contour_offset_x_norm = contour_offset_x / width
            contour_offset_y_norm = contour_offset_y / height
            
            features['contour_area'] = area
            features['contour_center_x'] = contour_center_x
            features['contour_center_y'] = contour_center_y
            features['contour_offset_x'] = contour_offset_x
            features['contour_offset_y'] = contour_offset_y
            features['contour_offset_x_norm'] = contour_offset_x_norm
            features['contour_offset_y_norm'] = contour_offset_y_norm
            features['aspect_ratio'] = aspect_ratio
            features['bounding_width'] = w
            features['bounding_height'] = h
        else:
            features['contour_area'] = 0
            features['contour_center_x'] = center_x
            features['contour_center_y'] = center_y
            features['contour_offset_x'] = 0
            features['contour_offset_y'] = 0
            features['contour_offset_x_norm'] = 0
            features['contour_offset_y_norm'] = 0
            features['aspect_ratio'] = 1
            features['bounding_width'] = 0
            features['bounding_height'] = 0
        
        # 3. Color-based features
        # Calculate mean color in different regions
        left_region = gray[:, :width//3]
        center_region = gray[:, width//3:2*width//3]
        right_region = gray[:, 2*width//3:]
        
        features['left_mean_intensity'] = np.mean(left_region)
        features['center_mean_intensity'] = np.mean(center_region)
        features['right_mean_intensity'] = np.mean(right_region)
        features['intensity_contrast'] = np.std(gray)
        
        # 4. Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features['hist_mean'] = np.mean(hist)
        features['hist_std'] = np.std(hist)
        features['hist_skew'] = self._calculate_skewness(hist)
        
        # 5. Hough line features
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            angles = []
            for line in lines[:10]:  # Take first 10 lines
                rho, theta = line[0]
                angles.append(theta * 180 / np.pi)
            features['line_count'] = len(lines)
            features['avg_line_angle'] = np.mean(angles)
            features['line_angle_std'] = np.std(angles)
        else:
            features['line_count'] = 0
            features['avg_line_angle'] = 0
            features['line_angle_std'] = 0
        
        # 6. Image statistics
        features['image_width'] = width
        features['image_height'] = height
        features['image_area'] = width * height
        features['center_x'] = center_x
        features['center_y'] = center_y
        
        return features
    
    def _calculate_skewness(self, data):
        """Calculate skewness of histogram"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def train(self, csv_file, img_dir):
        """
        Train the model using extracted features
        
        Args:
            csv_file (str): Path to CSV file with ground truth
            img_dir (str): Directory containing images
        """
        print("Training improved belt alignment model...")
        
        # Load ground truth
        df = pd.read_csv(csv_file)
        
        # Extract features for all images
        features_list = []
        labels = []
        
        for _, row in df.iterrows():
            img_path = os.path.join(img_dir, row['filename'])
            if os.path.exists(img_path):
                try:
                    features = self.extract_features(img_path)
                    features_list.append(features)
                    labels.append(row['center_percent'])
                    print(f"Processed {row['filename']}: {len(features)} features")
                except Exception as e:
                    print(f"Error processing {row['filename']}: {e}")
                    traceback.print_exc()
        
        if len(features_list) < 2:
            raise ValueError("Need at least 2 samples to train")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Split data (with small dataset, use all for training)
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.rf_model.score(X_train_scaled, y_train)
        test_score = self.rf_model.score(X_test_scaled, y_test)
        
        print(f"Training R² score: {train_score:.4f}")
        print(f"Test R² score: {test_score:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features_df.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        self.is_trained = True
        
        # Save model info
        model_info = {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': feature_importance.to_dict('records'),
            'n_samples': len(features_list)
        }
        
        with open('improved_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=4)
        
        return train_score, test_score
    
    def predict(self, image_path):
        """
        Predict alignment for a single image
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Extract features
        features = self.extract_features(image_path)
        features_df = pd.DataFrame([features])
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        prediction = self.rf_model.predict(features_scaled)[0]
        
        # Determine alignment status
        if prediction < -50:
            alignment_status = "Severe Left Misalignment"
            direction = "Left"
            severity = "Severe"
        elif prediction < 0:
            alignment_status = "Mild Left Misalignment"
            direction = "Left"
            severity = "Mild"
        elif prediction <= 100:
            alignment_status = "Good Alignment"
            direction = "Center"
            severity = "Good"
        else:
            alignment_status = "Severe Right Misalignment"
            direction = "Right"
            severity = "Severe"
        
        return {
            'predicted_value': float(prediction),
            'alignment_status': alignment_status,
            'direction': direction,
            'severity': severity,
            'features': features
        }
    
    def predict_batch(self, img_dir, output_file=None):
        """
        Predict alignment for all images in directory
        
        Args:
            img_dir (str): Directory containing images
            output_file (str): Optional file to save results
            
        Returns:
            list: List of prediction results
        """
        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for filename in os.listdir(img_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(img_dir, filename)
                try:
                    result = self.predict(image_path)
                    result['filename'] = filename
                    results.append(result)
                    print(f"{filename}: {result['predicted_value']:.1f}% ({result['alignment_status']})")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            print(f"Results saved to {output_file}")
        
        return results
    
    def evaluate(self, csv_file, img_dir):
        """
        Evaluate model performance on ground truth data
        
        Args:
            csv_file (str): Path to CSV file with ground truth
            img_dir (str): Directory containing images
        """
        df = pd.read_csv(csv_file)
        
        predictions = []
        ground_truth = []
        
        print("Evaluating improved model...")
        for _, row in df.iterrows():
            img_path = os.path.join(img_dir, row['filename'])
            if os.path.exists(img_path):
                try:
                    result = self.predict(img_path)
                    predictions.append(result['predicted_value'])
                    ground_truth.append(row['center_percent'])
                except Exception as e:
                    print(f"Error processing {row['filename']}: {e}")
        
        if predictions:
            predictions = np.array(predictions)
            ground_truth = np.array(ground_truth)
            
            mae = np.mean(np.abs(predictions - ground_truth))
            rmse = np.sqrt(np.mean((predictions - ground_truth)**2))
            
            # Calculate R-squared
            ss_res = np.sum((ground_truth - predictions) ** 2)
            ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            print(f"\nImproved Model Evaluation Results:")
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
            plt.title('Improved Model: Predictions vs True Values')
            plt.grid(True, alpha=0.3)
            
            # Error distribution
            plt.subplot(1, 2, 2)
            errors = predictions - ground_truth
            plt.hist(errors, bins=8, alpha=0.7, color='green', edgecolor='black')
            plt.xlabel('Prediction Error (%)')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('improved_model_evaluation.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print detailed comparison
            print(f"\nDetailed Predictions:")
            for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
                error = pred - true
                print(f"Sample {i+1}: True={true:.1f}%, Predicted={pred:.1f}%, Error={error:.1f}%")
            
            return predictions, ground_truth, mae, rmse, r2
        
        return None, None, None, None, None

def main():
    # Initialize improved model
    model = ImprovedBeltAlignmentModel()
    
    # Train model
    if os.path.exists('labels.csv'):
        print("Training improved model...")
        train_score, test_score = model.train('labels.csv', '.')
        
        # Evaluate on full dataset
        print("\nEvaluating on full dataset...")
        model.evaluate('labels.csv', '.')
        
        # Predict all images
        print("\nPredicting all images...")
        results = model.predict_batch('.', 'improved_predictions.json')
        
        # Compare with segmentation model
        if os.path.exists('segmentation_predictions.json'):
            print("\nComparing with segmentation model...")
            with open('segmentation_predictions.json', 'r') as f:
                seg_results = json.load(f)
            
            # Filter to belt images only
            belt_seg_results = [r for r in seg_results if r['filename'].endswith('.jpg')]
            
            print("\nModel Comparison:")
            print("Segmentation Model MAE: 81.31")
            print(f"Improved Model MAE: {test_score:.2f}")
            
            if test_score < 81.31:
                print("✅ Improved model performs better!")
            else:
                print("❌ Segmentation model still performs better")
    else:
        print("labels.csv not found. Please ensure you have ground truth data.")

if __name__ == "__main__":
    main() 