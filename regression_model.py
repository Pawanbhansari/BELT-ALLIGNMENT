import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os

class BeltAlignmentRegressor(nn.Module):
    """
    Regression model for predicting continuous belt alignment percentage
    Output: Single value representing misalignment percentage
    - Negative values = Left misalignment
    - Positive values = Right misalignment
    - 0 = Perfect alignment
    """
    
    def __init__(self, pretrained=True, model_name='resnet18'):
        super(BeltAlignmentRegressor, self).__init__()
        
        # Load pretrained backbone
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Regression head for continuous output
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single output for regression
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the regressor head"""
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.regressor(features)
        return output.squeeze()  # Remove extra dimension

class BeltAlignmentRegressionDataset(Dataset):
    """
    Dataset for regression-based belt alignment prediction
    """
    
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # Get the continuous alignment value
        alignment_value = self.labels_df.iloc[idx]['center_percent']
        
        if self.transform:
            image = self.transform(image)
        
        return image, alignment_value

def get_regression_transforms(img_size=224):
    """Get transforms for regression training"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class BeltAlignmentRegressionTrainer:
    """
    Trainer for regression-based belt alignment prediction
    """
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function for regression
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.float().to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.float().to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate additional metrics
        mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets))**2))
        
        return avg_loss, mae, rmse, predictions, targets
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"Starting regression training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, mae, rmse, predictions, targets = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model('best_regression_model.pth')
                print(f"New best validation loss: {val_loss:.4f}")
            
            # Early stopping
            if epoch > 10 and self._check_early_stopping():
                print("Early stopping triggered")
                break
        
        # Final evaluation
        self.evaluate_model(predictions, targets)
        self.plot_training_history()
    
    def _check_early_stopping(self, patience=10):
        """Check if early stopping should be triggered"""
        recent_losses = self.val_losses[-patience:]
        return all(loss >= self.best_val_loss for loss in recent_losses)
    
    def save_model(self, filename):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'train_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
        }, filename)
        print(f"Model saved to {filename}")
    
    def evaluate_model(self, predictions, targets):
        """Evaluate the model with detailed metrics"""
        print("\n" + "="*50)
        print("REGRESSION MODEL EVALUATION")
        print("="*50)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        mse = np.mean((predictions - targets)**2)
        
        # Calculate R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
        print(f"Mean Square Error (MSE): {mse:.2f}")
        print(f"R-squared (R²): {r2:.4f}")
        
        # Plot predictions vs targets
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(targets, predictions, alpha=0.7, color='blue')
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('True Alignment (%)')
        plt.ylabel('Predicted Alignment (%)')
        plt.title('Predictions vs True Values')
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(2, 2, 2)
        residuals = predictions - targets
        plt.scatter(predictions, residuals, alpha=0.7, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Alignment (%)')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # Error distribution
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=10, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # Training history
        plt.subplot(2, 2, 4)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regression_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed predictions
        print(f"\nDetailed Predictions:")
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            error = pred - target
            print(f"Sample {i+1}: True={target:.1f}%, Predicted={pred:.1f}%, Error={error:.1f}%")
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('regression_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_regression_data_loaders(csv_file, img_dir, batch_size=2, img_size=224, train_split=0.8):
    """Create train and validation data loaders for regression"""
    # Load dataset
    dataset = BeltAlignmentRegressionDataset(csv_file, img_dir, transform=None)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Apply transforms
    train_transform, val_transform = get_regression_transforms(img_size)
    
    # Create datasets with transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def main():
    # Configuration
    config = {
        'model_name': 'resnet18',
        'batch_size': 2,
        'img_size': 224,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'train_split': 0.8
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating regression data loaders...")
    train_loader, val_loader = create_regression_data_loaders(
        csv_file='labels.csv',
        img_dir='.',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        train_split=config['train_split']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating regression model...")
    model = BeltAlignmentRegressor(pretrained=True, model_name=config['model_name'])
    model.to(device)
    
    # Create trainer
    trainer = BeltAlignmentRegressionTrainer(model, train_loader, val_loader, device, config)
    
    # Train the model
    trainer.train(config['num_epochs'])
    
    # Save final model
    trainer.save_model('final_regression_model.pth')
    
    # Save configuration
    import json
    with open('regression_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\nRegression training completed!")
    print("Files saved:")
    print("- final_regression_model.pth: Final trained regression model")
    print("- best_regression_model.pth: Best model during training")
    print("- regression_config.json: Training configuration")
    print("- regression_evaluation.png: Detailed evaluation plots")
    print("- regression_training_history.png: Training history")

if __name__ == "__main__":
    main() 