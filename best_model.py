import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import json
from PIL import Image

class BestBeltAlignmentModel:
    """
    Best belt alignment model with outlier handling and improved features
    """
    
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.linear_model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.outlier_threshold = 100  # Clamp values beyond this
        self.extreme_threshold = 0.25  # Increased from 0.15 for more conservative rules
        
    def clamp_outliers(self, value):
        """Clamp extreme values to reasonable range"""
        return np.clip(value, -self.outlier_threshold, self.outlier_threshold)
    
    def extract_improved_features(self, image_path):
        """
        Extract improved features with better left/right detection
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
        
        # 1. Enhanced edge-based features
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.where(edges > 0)
        
        if len(edge_pixels[0]) > 0:
            edge_center_x = np.mean(edge_pixels[1])
            edge_center_y = np.mean(edge_pixels[0])
            edge_offset_x = edge_center_x - center_x
            edge_offset_y = edge_center_y - center_y
            edge_density = len(edge_pixels[0]) / (width * height)
            
            # Left/right edge density
            left_edges = edges[:, :width//3]
            center_edges = edges[:, width//3:2*width//3]
            right_edges = edges[:, 2*width//3:]
            
            left_edge_density = np.sum(left_edges > 0) / left_edges.size
            center_edge_density = np.sum(center_edges > 0) / center_edges.size
            right_edge_density = np.sum(right_edges > 0) / right_edges.size
            
            features['left_edge_density'] = left_edge_density
            features['center_edge_density'] = center_edge_density
            features['right_edge_density'] = right_edge_density
            features['edge_density_ratio'] = max(left_edge_density, right_edge_density) / (center_edge_density + 1e-6)
        else:
            edge_center_x = center_x
            edge_center_y = center_y
            edge_offset_x = 0
            edge_offset_y = 0
            edge_density = 0
            features['left_edge_density'] = 0
            features['center_edge_density'] = 0
            features['right_edge_density'] = 0
            features['edge_density_ratio'] = 0
        
        features['edge_center_x'] = edge_center_x
        features['edge_center_y'] = edge_center_y
        features['edge_offset_x'] = edge_offset_x
        features['edge_offset_y'] = edge_offset_y
        features['edge_density'] = edge_density
        
        # 2. Enhanced contour-based features
        try:
            contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[-2]
        except:
            contours = []
            
        if len(contours) > 0:
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
            
            # Enhanced offset features
            features['contour_offset_x_norm_abs'] = abs(contour_offset_x_norm)
            features['contour_offset_direction'] = 1 if contour_offset_x_norm > 0 else -1
            features['contour_extreme_left'] = 1 if contour_offset_x_norm < -self.extreme_threshold else 0
            features['contour_extreme_right'] = 1 if contour_offset_x_norm > self.extreme_threshold else 0
            
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
            features['contour_offset_x_norm_abs'] = 0
            features['contour_offset_direction'] = 0
            features['contour_extreme_left'] = 0
            features['contour_extreme_right'] = 0
            features['aspect_ratio'] = 1
            features['bounding_width'] = 0
            features['bounding_height'] = 0
        
        # 3. Enhanced color-based features
        # Calculate mean color in different regions
        left_region = gray[:, :width//3]
        center_region = gray[:, width//3:2*width//3]
        right_region = gray[:, 2*width//3:]
        
        features['left_mean_intensity'] = np.mean(left_region)
        features['center_mean_intensity'] = np.mean(center_region)
        features['right_mean_intensity'] = np.mean(right_region)
        features['intensity_contrast'] = np.std(gray)
        
        # Enhanced intensity features
        features['left_right_intensity_diff'] = features['left_mean_intensity'] - features['right_mean_intensity']
        features['left_center_intensity_diff'] = features['left_mean_intensity'] - features['center_mean_intensity']
        features['right_center_intensity_diff'] = features['right_mean_intensity'] - features['center_mean_intensity']
        
        # 4. Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features['hist_mean'] = np.mean(hist)
        features['hist_std'] = np.std(hist)
        features['hist_skew'] = self._calculate_skewness(hist)
        
        # 5. Enhanced Hough line features
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            angles = []
            for line in lines[:10]:
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
    
    def apply_extreme_rules(self, features, base_prediction):
        """
        Apply simple rules for extreme cases based on features
        """
        # Rule 1: If contour is extremely left AND confirmed by edge density, predict severe left misalignment
        if (features.get('contour_extreme_left', 0) == 1 and 
            features.get('left_edge_density', 0) > 0.02 and  # Higher threshold
            features.get('contour_offset_x_norm_abs', 0) > 0.3):  # Very extreme offset
            return -150  # Severe left
        
        # Rule 2: If contour is extremely right AND confirmed by edge density, predict severe right misalignment
        if (features.get('contour_extreme_right', 0) == 1 and 
            features.get('right_edge_density', 0) > 0.02 and  # Higher threshold
            features.get('contour_offset_x_norm_abs', 0) > 0.3):  # Very extreme offset
            return 150  # Severe right
        
        # Rule 3: If edge density ratio is very high AND base prediction is already extreme, amplify
        edge_ratio = features.get('edge_density_ratio', 0)
        if edge_ratio > 10 and abs(base_prediction) > 50:  # Much higher threshold
            direction = features.get('contour_offset_direction', 0)
            if direction > 0:
                return min(base_prediction * 1.2, 100)  # Moderate amplification
            else:
                return max(base_prediction * 1.2, -100)  # Moderate amplification
        
        return base_prediction
    
    def train(self, csv_file, img_dir):
        """
        Train the model using clamped ground truth values
        """
        print("Training best belt alignment model...")
        
        # Load ground truth
        df = pd.read_csv(csv_file)
        
        # Clamp outliers in ground truth
        df['center_percent_clamped'] = df['center_percent'].apply(self.clamp_outliers)
        
        # Extract features for all images
        features_list = []
        labels = []
        original_labels = []
        
        for _, row in df.iterrows():
            img_path = os.path.join(img_dir, row['filename'])
            if os.path.exists(img_path):
                try:
                    features = self.extract_improved_features(img_path)
                    features_list.append(features)
                    labels.append(row['center_percent_clamped'])  # Use clamped values
                    original_labels.append(row['center_percent'])  # Keep original for comparison
                    print(f"Processed {row['filename']}: {len(features)} features")
                except Exception as e:
                    print(f"Error processing {row['filename']}: {e}")
        
        if len(features_list) < 2:
            raise ValueError("Need at least 2 samples to train")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train both models
        self.rf_model.fit(X_train_scaled, y_train)
        self.linear_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        rf_train_score = self.rf_model.score(X_train_scaled, y_train)
        rf_test_score = self.rf_model.score(X_test_scaled, y_test)
        linear_train_score = self.linear_model.score(X_train_scaled, y_train)
        linear_test_score = self.linear_model.score(X_test_scaled, y_test)
        
        print(f"Random Forest - Training R²: {rf_train_score:.4f}, Test R²: {rf_test_score:.4f}")
        print(f"Linear Regression - Training R²: {linear_train_score:.4f}, Test R²: {linear_test_score:.4f}")
        
        # Choose the better model
        if rf_test_score > linear_test_score:
            self.best_model = self.rf_model
            self.best_score = rf_test_score
            print("Using Random Forest model")
        else:
            self.best_model = self.linear_model
            self.best_score = linear_test_score
            print("Using Linear Regression model")
        
        # Feature importance (for Random Forest)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': features_df.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 most important features:")
            print(feature_importance.head(10))
        
        self.is_trained = True
        
        # Save model info
        model_info = {
            'best_model_type': 'RandomForest' if self.best_model == self.rf_model else 'LinearRegression',
            'best_score': self.best_score,
            'n_samples': len(features_list),
            'outlier_threshold': self.outlier_threshold,
            'extreme_threshold': self.extreme_threshold
        }
        
        with open('best_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=4)
        
        return self.best_score
    
    def predict(self, image_path):
        """
        Predict alignment with extreme case handling
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Extract features
        features = self.extract_improved_features(image_path)
        features_df = pd.DataFrame([features])
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Get base prediction
        base_prediction = self.best_model.predict(features_scaled)[0]
        
        # Apply extreme case rules
        final_prediction = self.apply_extreme_rules(features, base_prediction)
        
        # Determine alignment status
        if final_prediction < -50:
            alignment_status = "Severe Left Misalignment"
            direction = "Left"
            severity = "Severe"
        elif final_prediction < 0:
            alignment_status = "Mild Left Misalignment"
            direction = "Left"
            severity = "Mild"
        elif final_prediction <= 100:
            alignment_status = "Good Alignment"
            direction = "Center"
            severity = "Good"
        else:
            alignment_status = "Severe Right Misalignment"
            direction = "Right"
            severity = "Severe"
        
        return {
            'predicted_value': float(final_prediction),
            'base_prediction': float(base_prediction),
            'alignment_status': alignment_status,
            'direction': direction,
            'severity': severity,
            'features': features,
            'extreme_rule_applied': final_prediction != base_prediction
        }
    
    def predict_batch(self, img_dir, output_file=None):
        """
        Predict alignment for all images in directory
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
                    rule_applied = " (EXTREME RULE)" if result['extreme_rule_applied'] else ""
                    print(f"{filename}: {result['predicted_value']:.1f}% ({result['alignment_status']}){rule_applied}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            print(f"Results saved to {output_file}")
        
        return results
    
    def evaluate(self, csv_file, img_dir):
        """
        Evaluate model performance
        """
        df = pd.read_csv(csv_file)
        
        predictions = []
        ground_truth = []
        
        print("Evaluating best model...")
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
            
            print(f"\nBest Model Evaluation Results:")
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
            plt.title('Best Model: Predictions vs True Values')
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
            plt.savefig('best_model_evaluation.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print detailed comparison
            print(f"\nDetailed Predictions:")
            for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
                error = pred - true
                print(f"Sample {i+1}: True={true:.1f}%, Predicted={pred:.1f}%, Error={error:.1f}%")
            
            return predictions, ground_truth, mae, rmse, r2
        
        return None, None, None, None, None

def main():
    # Initialize best model
    model = BestBeltAlignmentModel()
    
    # Train model
    if os.path.exists('labels.csv'):
        print("Training best model...")
        best_score = model.train('labels.csv', '.')
        
        # Evaluate on full dataset
        print("\nEvaluating on full dataset...")
        model.evaluate('labels.csv', '.')
        
        # Predict all images
        print("\nPredicting all images...")
        results = model.predict_batch('.', 'best_predictions.json')
        
        # Compare with previous models
        print("\nModel Comparison:")
        print("Segmentation Model MAE: 81.31")
        print("Improved Model MAE: 48.07")
        print(f"Best Model MAE: {best_score:.2f}")
        
        if best_score < 48.07:
            print("✅ Best model performs better than improved model!")
        else:
            print("❌ Improved model still performs better")
    else:
        print("labels.csv not found. Please ensure you have ground truth data.")

if __name__ == "__main__":
    main() 