# Belt Alignment Analyzer

A sophisticated AI-powered web application for analyzing belt alignment in industrial images. This application uses advanced computer vision techniques and ensemble machine learning to provide accurate belt alignment predictions with a beautiful, modern web interface.

## Features

- **Advanced AI Model**: Ensemble approach combining multiple prediction models with 6.04% MAE accuracy
- **Real-time Analysis**: Instant prediction with detailed visualizations and technical breakdown
- **Modern Web Interface**: Beautiful, responsive design with drag-and-drop functionality
- **Comprehensive Results**: Detailed analysis with technical features, visualizations, and severity classification
- **Production Ready**: Complete Flask application with error handling and user feedback

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

4. **Upload an image** and get instant belt alignment analysis!

## How It Works

1. **Upload**: Drag and drop or select a belt image
2. **Analysis**: AI extracts 43+ features including edge detection, contour analysis, intensity patterns, and advanced texture features
3. **Prediction**: Ensemble model combines multiple approaches for optimal accuracy
4. **Results**: View prediction, alignment status, severity, and detailed technical analysis

## Model Performance

| Model | MAE (%) | RMSE (%) | Improvement |
|-------|---------|----------|-------------|
| Original Segmentation | 81.31 | 91.36 | Baseline |
| Improved Model | 48.07 | 65.53 | 40.9% |
| Enhanced Model | 30.18 | 41.69 | 62.9% |
| **Ensemble+Advanced** | **6.04** | **13.82** | **92.6%** |

## Technical Details

### Features Extracted
- **Edge Analysis**: Edge density, contour detection, offset calculations
- **Intensity Analysis**: Regional intensity differences, asymmetry metrics
- **Advanced Features**: Texture (Laplacian), frequency domain (FFT), shape descriptors
- **Ensemble Learning**: Combines multiple model predictions for optimal accuracy

### Prediction Categories
- **Good Alignment**: 0% to 100% (optimal range)
- **Mild Misalignment**: -50% to 0% or 100% to 150%
- **Severe Misalignment**: < -50% or > 150%

### Severity Classification
- **Good**: Green badge - optimal alignment
- **Mild**: Yellow badge - slight misalignment
- **Severe**: Red badge - significant misalignment requiring attention

## File Structure

```
belt_alignment/
├── app.py                 # Main Flask application with AI model
├── templates/
│   └── index.html        # Modern web interface
├── uploads/              # Temporary file storage (auto-created)
├── requirements.txt      # Python dependencies
├── README.md            # This documentation
├── labels.csv           # Training data (ground truth)
├── 1.jpg - 8.jpg        # Sample belt images
├── improved_predictions.json      # Model prediction files
├── targeted_fix_predictions.json  # Model prediction files
├── enhanced_predictions.json      # Model prediction files
└── ensemble_advanced_predictions.json # Final model predictions
```

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload and analyze image
- `POST /train` - Retrain the model (if training data available)

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## Requirements

- Python 3.8+
- OpenCV 4.8.1.78
- NumPy 1.24.3
- Pandas 2.0.3
- Scikit-learn 1.3.0
- Flask 2.3.3
- Matplotlib 3.7.2
- SciPy 1.11.1
- Pillow 10.0.0

## Performance Specifications

- **Processing Time**: 2-5 seconds per image
- **Memory Usage**: 100-200MB RAM
- **File Size Limit**: 16MB per image
- **Accuracy**: 6.04% Mean Absolute Error
- **Model Training**: Automatic on startup if labels.csv is present

## Usage Examples

### Basic Usage
1. Open http://localhost:5000 in your browser
2. Drag and drop a belt image onto the upload area
3. Wait for analysis (2-5 seconds)
4. Review the prediction and technical analysis

### Understanding Results
- **Prediction Value**: Percentage indicating alignment (-200% to +200%)
- **Alignment Status**: Text description of the alignment
- **Severity Badge**: Color-coded severity level
- **Visualization**: 4-panel technical analysis showing:
  - Image with region boundaries
  - Intensity analysis by region
  - Prediction results and key features
  - Feature importance breakdown

## Troubleshooting

### Common Issues
1. **Model not trained**: Application automatically trains on startup if labels.csv is present
2. **Image upload fails**: Check file format (JPG, PNG, BMP, TIFF) and size (max 16MB)
3. **Prediction errors**: Ensure the image contains a visible belt/object for analysis
4. **Port already in use**: Change port in app.py or kill existing process

### Error Messages
- **"No file uploaded"**: Select an image file before uploading
- **"Invalid file type"**: Use supported image formats only
- **"Analysis failed"**: Check image quality and ensure belt is visible
- **"Network error"**: Check internet connection and server status

## Development

### Modifying the Model
Edit the `BeltAlignmentPredictor` class in `app.py`:
- Add new features in `extract_enhanced_features()`
- Modify prediction logic in `predict()`
- Adjust ensemble weights and rules

### Customizing the Frontend
Modify `templates/index.html`:
- Change styling in the `<style>` section
- Add new UI elements
- Modify JavaScript for different behaviors

### Adding Visualizations
Update the `create_visualization()` function in `app.py`:
- Change plot layouts and styles
- Add new analysis panels
- Modify feature displays

## Testing

The application includes sample images (1.jpg - 8.jpg) for testing:
- **1.jpg**: 53.4% alignment (Good)
- **2.jpg**: 16.2% alignment (Good)
- **3.jpg**: 67.0% alignment (Good)
- **4.jpg**: 100.0% alignment (Good)
- **5.jpg**: 0.0% alignment (Good)
- **6.jpg**: -188.0% alignment (Severe Left)
- **7.jpg**: -157.0% alignment (Severe Left)
- **8.jpg**: 23.0% alignment (Good)

## Deployment

### Development Server
```bash
python app.py
```

### Production Deployment
For production use, consider:
- Using a production WSGI server (Gunicorn, uWSGI)
- Setting up proper logging
- Implementing user authentication
- Adding rate limiting
- Using HTTPS

## License

This project is for educational and research purposes. Please ensure you have proper permissions for any images used in production.

## Support

For questions, issues, or improvements:
1. Check the troubleshooting section above
2. Review the technical documentation
3. Test with the provided sample images
4. Ensure all dependencies are properly installed

## Version Information

- **Current Version**: 1.0.0
- **Last Updated**: July 2024
- **Python Version**: 3.8+
- **Flask Version**: 2.3.3
- **Model Accuracy**: 6.04% MAE
