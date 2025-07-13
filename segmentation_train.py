import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import cv2
from tqdm import tqdm

from segmentation_model import BeltSegmentationUNet

class BeltSegmentationDataset(Dataset):
    """
    Dataset for belt segmentation with automatic mask generation
    """
    
    def __init__(self, csv_file, img_dir, transform=None, img_size=224):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        original_size = image.size
        
        # Get alignment value for mask generation
        alignment_value = self.labels_df.iloc[idx]['center_percent']
        
        # Create segmentation mask based on alignment
        mask = self.create_belt_mask(alignment_value, original_size)
        
        if self.transform:
            image = self.transform(image)
            # Resize mask to match image
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def create_belt_mask(self, center_percent, img_size):
        """
        Create realistic belt segmentation mask based on alignment percentage
        """
        width, height = img_size
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Belt parameters
        belt_height = height // 6  # Belt thickness
        belt_start = (height - belt_height) // 2
        
        # Calculate belt center based on alignment percentage
        # Normalize center_percent to image width
        max_offset = width * 0.4  # Maximum 40% of image width
        offset = (center_percent / 100) * max_offset
        belt_center = width // 2 + int(offset)
        
        # Belt width (varies based on alignment severity)
        base_width = width // 4
        if abs(center_percent) > 50:
            belt_width = base_width // 2  # Narrower for severe misalignment
        else:
            belt_width = base_width
        
        # Ensure belt stays within image bounds
        belt_left = max(0, belt_center - belt_width // 2)
        belt_right = min(width, belt_center + belt_width // 2)
        if belt_right > belt_left:
            belt_region = np.ones((belt_height, belt_right - belt_left), dtype=np.uint8)
            # Add some noise to make it more realistic
            noise = np.random.random(belt_region.shape) > 0.1
            belt_region = belt_region * noise
            mask[belt_start:belt_start+belt_height, belt_left:belt_right] = belt_region
        # else: do not draw belt if region is invalid
        return mask

def get_segmentation_transforms(img_size=224):
    """Get transforms for segmentation training"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class SegmentationTrainer:
    """
    Trainer for belt segmentation model
    """
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function for segmentation (Dice + CrossEntropy)
        self.criterion = CombinedLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_ious = []
        self.val_ious = []
        self.best_val_iou = 0.0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_iou = 0.0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc='Training')):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate IoU
            pred_mask = torch.argmax(output, dim=1)
            iou = self.calculate_iou(pred_mask, target)
            
            total_loss += loss.item()
            total_iou += iou.item()
        
        return total_loss / len(self.train_loader), total_iou / len(self.train_loader)
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Calculate IoU
                pred_mask = torch.argmax(output, dim=1)
                iou = self.calculate_iou(pred_mask, target)
                
                total_loss += loss.item()
                total_iou += iou.item()
                
                all_predictions.extend(pred_mask.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        return (total_loss / len(self.val_loader), 
                total_iou / len(self.val_loader), 
                all_predictions, all_targets)
    
    def calculate_iou(self, pred, target):
        """Calculate IoU for segmentation"""
        intersection = (pred == target).sum()
        union = pred.numel()
        return intersection.float() / union
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"Starting segmentation training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_iou = self.train_epoch()
            
            # Validation
            val_loss, val_iou, predictions, targets = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_iou)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_ious.append(train_iou)
            self.val_ious.append(val_iou)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            # Save best model
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                self.save_model('best_segmentation_model.pth')
                print(f"New best validation IoU: {val_iou:.4f}")
            
            # Early stopping
            if epoch > 10 and self._check_early_stopping():
                print("Early stopping triggered")
                break
        
        # Final evaluation
        self.evaluate_model(predictions, targets)
        self.plot_training_history()
    
    def _check_early_stopping(self, patience=10):
        """Check if early stopping should be triggered"""
        recent_ious = self.val_ious[-patience:]
        return all(iou <= self.best_val_iou for iou in recent_ious)
    
    def save_model(self, filename):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_iou': self.best_val_iou,
            'train_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_ious': self.train_ious,
                'val_ious': self.val_ious
            }
        }, filename)
        print(f"Model saved to {filename}")
    
    def evaluate_model(self, predictions, targets):
        """Evaluate the model with detailed metrics"""
        print("\n" + "="*50)
        print("SEGMENTATION MODEL EVALUATION")
        print("="*50)
        
        # Calculate overall IoU
        total_iou = 0
        total_pixels = 0
        
        for pred, target in zip(predictions, targets):
            intersection = (pred == target).sum()
            total_pixels += pred.size
            total_iou += intersection
        
        overall_iou = total_iou / total_pixels
        print(f"Overall IoU: {overall_iou:.4f}")
        
        # Plot sample predictions
        self.plot_sample_predictions(predictions[:4], targets[:4])
    
    def plot_sample_predictions(self, predictions, targets):
        """Plot sample segmentation results"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i in range(min(4, len(predictions))):
            # Original mask
            axes[0, i].imshow(targets[i], cmap='gray')
            axes[0, i].set_title(f'Ground Truth {i+1}')
            axes[0, i].axis('off')
            
            # Predicted mask
            axes[1, i].imshow(predictions[i], cmap='gray')
            axes[1, i].set_title(f'Predicted {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('segmentation_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # IoU plot
        ax2.plot(self.train_ious, label='Train IoU')
        ax2.plot(self.val_ious, label='Validation IoU')
        ax2.set_title('Training and Validation IoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('segmentation_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

class CombinedLoss(nn.Module):
    """
    Combined loss function for segmentation (Dice + CrossEntropy)
    """
    
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, output, target):
        # Cross-entropy loss
        ce_loss = self.ce_loss(output, target)
        
        # Dice loss
        dice_loss = self.dice_loss(output, target)
        
        # Combined loss
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return total_loss
    
    def dice_loss(self, output, target):
        """Calculate Dice loss"""
        smooth = 1e-6
        
        # Convert to one-hot encoding
        num_classes = output.size(1)
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Softmax for output
        output_softmax = F.softmax(output, dim=1)
        
        # Calculate Dice coefficient
        intersection = (output_softmax * target_onehot).sum(dim=(2, 3))
        union = output_softmax.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss

def create_segmentation_data_loaders(csv_file, img_dir, batch_size=2, img_size=224, train_split=0.8):
    """Create train and validation data loaders for segmentation"""
    # Load dataset
    dataset = BeltSegmentationDataset(csv_file, img_dir, transform=None, img_size=img_size)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Apply transforms
    train_transform, val_transform = get_segmentation_transforms(img_size)
    
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
    print("Creating segmentation data loaders...")
    train_loader, val_loader = create_segmentation_data_loaders(
        csv_file='labels.csv',
        img_dir='.',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        train_split=config['train_split']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating segmentation model...")
    model = BeltSegmentationUNet(num_classes=2, pretrained=True)  # Binary segmentation
    model.to(device)
    
    # Create trainer
    trainer = SegmentationTrainer(model, train_loader, val_loader, device, config)
    
    # Train the model
    trainer.train(config['num_epochs'])
    
    # Save final model
    trainer.save_model('final_segmentation_model.pth')
    
    # Save configuration
    import json
    with open('segmentation_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\nSegmentation training completed!")
    print("Files saved:")
    print("- final_segmentation_model.pth: Final trained segmentation model")
    print("- best_segmentation_model.pth: Best model during training")
    print("- segmentation_config.json: Training configuration")
    print("- segmentation_samples.png: Sample predictions")
    print("- segmentation_training_history.png: Training history")

if __name__ == "__main__":
    main() 