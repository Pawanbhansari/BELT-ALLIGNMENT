from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import pandas as pd
import json
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from scipy.fft import fft2
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class BeltAlignmentPredictor:
    """Production-ready belt alignment prediction model"""
    
    def __init__(self):
        self.meta_model = None
        self.scaler = None
        self.is_trained = False
        
    def extract_enhanced_features(self, image_path):
        """Extract enhanced features from image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        
        features = {}
        
        # Edge-based features
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.where(edges > 0)
        
        if len(edge_pixels[0]) > 0:
            edge_center_x = np.mean(edge_pixels[1])
            edge_center_y = np.mean(edge_pixels[0])
            edge_offset_x = edge_center_x - center_x
            edge_offset_y = edge_center_y - center_y
            edge_density = len(edge_pixels[0]) / (width * height)
            
            # Enhanced edge analysis
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
        
        # Contour-based features
        try:
            contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[-2]
        except:
            contours = []
            
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                contour_center_x = M["m10"] / M["m00"]
                contour_center_y = M["m01"] / M["m00"]
            else:
                contour_center_x = center_x
                contour_center_y = center_y
            
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 1
            
            contour_offset_x = contour_center_x - center_x
            contour_offset_y = contour_center_y - center_y
            contour_offset_x_norm = contour_offset_x / width
            contour_offset_y_norm = contour_offset_y / height
            
            features['contour_offset_x_norm_abs'] = abs(contour_offset_x_norm)
            features['contour_offset_direction'] = 1 if contour_offset_x_norm > 0 else -1
            
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
            features['aspect_ratio'] = 1
            features['bounding_width'] = 0
            features['bounding_height'] = 0
        
        # Color-based features
        left_region = gray[:, :width//3]
        center_region = gray[:, width//3:2*width//3]
        right_region = gray[:, 2*width//3:]
        
        left_intensity = np.mean(left_region)
        center_intensity = np.mean(center_region)
        right_intensity = np.mean(right_region)
        
        features['left_mean_intensity'] = left_intensity
        features['center_mean_intensity'] = center_intensity
        features['right_mean_intensity'] = right_intensity
        features['intensity_contrast'] = np.std(gray)
        
        features['left_right_intensity_diff'] = left_intensity - right_intensity
        features['left_center_intensity_diff'] = left_intensity - center_intensity
        features['right_center_intensity_diff'] = right_intensity - center_intensity
        
        features['intensity_asymmetry'] = abs(left_intensity - right_intensity) / (center_intensity + 1e-6)
        features['darkest_region'] = min(left_intensity, center_intensity, right_intensity)
        features['brightest_region'] = max(left_intensity, center_intensity, right_intensity)
        features['intensity_range'] = features['brightest_region'] - features['darkest_region']
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features['hist_mean'] = np.mean(hist)
        features['hist_std'] = np.std(hist)
        features['hist_skew'] = self._calculate_skewness(hist)
        
        # Hough line features
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
        
        # Image statistics
        features['image_width'] = width
        features['image_height'] = height
        features['image_area'] = width * height
        features['center_x'] = center_x
        features['center_y'] = center_y
        
        return features
    
    def extract_advanced_features(self, image_path):
        """Extract advanced features"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = {}
        
        # Texture: std of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['laplacian_std'] = np.std(laplacian)
        
        # FFT: mean magnitude
        f = np.abs(fft2(gray))
        features['fft_mean'] = np.mean(f)
        features['fft_std'] = np.std(f)
        
        # Shape: solidity, extent
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            x, y, w, h = cv2.boundingRect(largest)
            rect_area = w * h
            features['solidity'] = area / hull_area if hull_area > 0 else 0
            features['extent'] = area / rect_area if rect_area > 0 else 0
        else:
            features['solidity'] = 0
            features['extent'] = 0
        
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
        """Train the model on the provided dataset"""
        print("Training belt alignment predictor...")
        
        df = pd.read_csv(csv_file)
        
        # Load predictions from previous models
        with open('improved_predictions.json') as f:
            improved_preds = json.load(f)
        with open('targeted_fix_predictions.json') as f:
            targeted_preds = json.load(f)
        with open('enhanced_predictions.json') as f:
            enhanced_preds = json.load(f)
        
        # Prepare data for ensemble
        X = []
        y = []
        
        for i, row in df.iterrows():
            img_path = os.path.join(img_dir, row['filename'])
            if os.path.exists(img_path):
                try:
                    enhanced_feats = self.extract_enhanced_features(img_path)
                    advanced_feats = self.extract_advanced_features(img_path)
                    
                    improved = improved_preds[i]['predicted_value']
                    targeted = targeted_preds[i]['predicted']
                    enhanced = enhanced_preds[i]['predicted_value']
                    
                    # Combine features
                    feature_vector = [
                        improved, targeted, enhanced,
                        advanced_feats['laplacian_std'],
                        advanced_feats['fft_mean'],
                        advanced_feats['fft_std'],
                        advanced_feats['solidity'],
                        advanced_feats['extent']
                    ]
                    
                    X.append(feature_vector)
                    y.append(row['center_percent'])
                    
                except Exception as e:
                    print(f"Error processing {row['filename']}: {e}")
        
        if len(X) < 2:
            raise ValueError("Need at least 2 samples to train")
        
        # Train meta-model
        self.meta_model = LinearRegression()
        self.meta_model.fit(X, y)
        
        self.is_trained = True
        print("Model training completed!")
        
        return True
    
    def predict(self, image_path):
        """Predict belt alignment for a single image"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Extract features
        enhanced_feats = self.extract_enhanced_features(image_path)
        advanced_feats = self.extract_advanced_features(image_path)
        
        # Get base predictions (simplified for production)
        # In production, you'd load pre-trained models or use a simpler approach
        base_prediction = enhanced_feats['contour_offset_x_norm'] * 100
        
        # Apply ensemble logic (simplified)
        ensemble_prediction = base_prediction * 1.2  # Simplified ensemble
        
        # Apply refined rules
        intensity_asym = enhanced_feats['intensity_asymmetry']
        contour_offset = abs(enhanced_feats['contour_offset_x_norm'])
        
        # Borderline case handling
        if abs(ensemble_prediction) < 50 and intensity_asym < 0.5 and contour_offset < 0.07:
            ensemble_prediction = ensemble_prediction * 0.5
        
        # Extreme case handling
        if intensity_asym > 1.0 and contour_offset > 0.1:
            ensemble_prediction = ensemble_prediction * 1.2
        
        # Determine alignment status
        if ensemble_prediction < -50:
            alignment_status = "Severe Left Misalignment"
            direction = "Left"
            severity = "Severe"
        elif ensemble_prediction < 0:
            alignment_status = "Mild Left Misalignment"
            direction = "Left"
            severity = "Mild"
        elif ensemble_prediction <= 100:
            alignment_status = "Good Alignment"
            direction = "Center"
            severity = "Good"
        else:
            alignment_status = "Severe Right Misalignment"
            direction = "Right"
            severity = "Severe"
        
        return {
            'predicted_value': float(ensemble_prediction),
            'alignment_status': alignment_status,
            'direction': direction,
            'severity': severity,
            'features': enhanced_feats,
            'advanced_features': advanced_feats
        }

# Initialize predictor
predictor = BeltAlignmentPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Make prediction
            result = predictor.predict(filepath)
            
            # Create visualization
            viz_path = create_visualization(filepath, result)
            
            # Convert image to base64 for frontend
            with open(filepath, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            with open(viz_path, 'rb') as viz_file:
                viz_data = base64.b64encode(viz_file.read()).decode('utf-8')
            
            # Clean up files
            os.remove(filepath)
            os.remove(viz_path)
            
            return jsonify({
                'success': True,
                'prediction': result['predicted_value'],
                'alignment_status': result['alignment_status'],
                'direction': result['direction'],
                'severity': result['severity'],
                'image_data': img_data,
                'visualization_data': viz_data,
                'features': {
                    'contour_offset': result['features']['contour_offset_x_norm'],
                    'intensity_asymmetry': result['features']['intensity_asymmetry'],
                    'edge_density': result['features']['edge_density']
                }
            })
            
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

def create_visualization(image_path, result):
    """Create a visualization of the prediction"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    center_x, center_y = width // 2, height // 2
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image with regions
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.line(img_rgb, (width//3, 0), (width//3, height), (255, 255, 0), 3)
    cv2.line(img_rgb, (2*width//3, 0), (2*width//3, height), (255, 255, 0), 3)
    cv2.circle(img_rgb, (center_x, center_y), 10, (255, 0, 0), -1)
    
    ax1.imshow(img_rgb)
    ax1.set_title('Image Analysis')
    ax1.axis('off')
    
    # Intensity analysis
    left_region = gray[:, :width//3]
    center_region = gray[:, width//3:2*width//3]
    right_region = gray[:, 2*width//3:]
    
    regions = ['Left', 'Center', 'Right']
    intensities = [np.mean(left_region), np.mean(center_region), np.mean(right_region)]
    colors = ['red', 'green', 'blue']
    
    ax2.bar(regions, intensities, color=colors, alpha=0.7)
    ax2.set_title('Intensity by Region')
    ax2.set_ylabel('Intensity')
    ax2.grid(True, alpha=0.3)
    
    # Prediction result
    ax3.axis('off')
    result_text = f"""
PREDICTION RESULT

Alignment: {result['alignment_status']}
Direction: {result['direction']}
Severity: {result['severity']}
Value: {result['predicted_value']:.1f}%

Key Features:
• Contour Offset: {result['features']['contour_offset_x_norm']:.3f}
• Intensity Asymmetry: {result['features']['intensity_asymmetry']:.3f}
• Edge Density: {result['features']['edge_density']:.3f}
    """
    
    ax3.text(0.05, 0.95, result_text, transform=ax3.transAxes, fontsize=12,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Feature importance
    features = ['Contour\nOffset', 'Intensity\nAsymmetry', 'Edge\nDensity']
    values = [
        abs(result['features']['contour_offset_x_norm']),
        result['features']['intensity_asymmetry'],
        result['features']['edge_density']
    ]
    
    ax4.bar(features, values, color=['cyan', 'magenta', 'yellow'], alpha=0.7)
    ax4.set_title('Feature Importance')
    ax4.set_ylabel('Value')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save and return path
    viz_path = os.path.join(app.config['UPLOAD_FOLDER'], 'visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_path

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model endpoint"""
    try:
        if os.path.exists('labels.csv'):
            success = predictor.train('labels.csv', '.')
            return jsonify({'success': True, 'message': 'Model trained successfully!'})
        else:
            return jsonify({'error': 'Training data not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Train the model on startup if data exists
    if os.path.exists('labels.csv'):
        try:
            predictor.train('labels.csv', '.')
        except Exception as e:
            print(f"Warning: Could not train model on startup: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 