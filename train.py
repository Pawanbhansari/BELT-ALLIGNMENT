import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data_loader import create_data_loaders, BeltAlignmentDataset, get_transforms
from model import get_model, count_parameters

class BeltAlignmentTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function
        if config['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min' if config['task'] == 'regression' else 'max',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            if self.config['task'] == 'classification':
                loss = self.criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            else:
                loss = self.criterion(output, target.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            if self.config['task'] == 'classification':
                accuracy = 100. * correct / total
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{accuracy:.2f}%'})
            else:
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total if self.config['task'] == 'classification' else 0.0
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                if self.config['task'] == 'classification':
                    loss = self.criterion(output, target)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                    all_predictions.extend(pred.cpu().numpy().flatten())
                    all_targets.extend(target.cpu().numpy())
                else:
                    loss = self.criterion(output, target.float())
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total if self.config['task'] == 'classification' else 0.0
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, predictions, targets = self.validate_epoch()
            
            # Update learning rate
            if self.config['task'] == 'classification':
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print results
            if self.config['task'] == 'classification':
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if self.config['task'] == 'classification':
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.save_model('best_model.pth')
                    print(f"New best validation accuracy: {val_acc:.2f}%")
            else:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model('best_model.pth')
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
        if self.config['task'] == 'classification':
            recent_accuracies = self.val_accuracies[-patience:]
            return all(acc <= self.best_val_accuracy for acc in recent_accuracies)
        else:
            recent_losses = self.val_losses[-patience:]
            return all(loss >= self.best_val_loss for loss in recent_losses)
    
    def save_model(self, filename):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_loss': self.best_val_loss,
            'train_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies
            }
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load the model"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if 'train_history' in checkpoint:
            history = checkpoint['train_history']
            self.train_losses = history.get('train_losses', [])
            self.val_losses = history.get('val_losses', [])
            self.train_accuracies = history.get('train_accuracies', [])
            self.val_accuracies = history.get('val_accuracies', [])
        
        print(f"Model loaded from {filename}")
    
    def evaluate_model(self, predictions, targets):
        """Evaluate the model with detailed metrics"""
        if self.config['task'] == 'classification':
            print("\n" + "="*50)
            print("MODEL EVALUATION")
            print("="*50)
            
            # Classification report
            class_names = ['Severe Left', 'Mild Left', 'Good', 'Severe Right']
            print("\nClassification Report:")
            print(classification_report(targets, predictions, target_names=class_names, labels=np.arange(len(class_names))))
            
            # Confusion matrix
            cm = confusion_matrix(targets, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Overall accuracy
            accuracy = accuracy_score(targets, predictions)
            print(f"\nOverall Accuracy: {accuracy:.4f}")
    
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
        
        # Accuracy plot (for classification)
        if self.config['task'] == 'classification':
            ax2.plot(self.train_accuracies, label='Train Accuracy')
            ax2.plot(self.val_accuracies, label='Validation Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Configuration
    config = {
        'task': 'classification',  # 'classification' or 'regression'
        'model_type': 'classifier',  # 'classifier', 'regressor', or 'attention'
        'model_name': 'resnet18',  # 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'
        'num_classes': 4,
        'batch_size': 2,  # Small batch size for small dataset
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
    print("Creating data loaders...")
    train_loader, val_loader, class_names = create_data_loaders(
        csv_file='labels.csv',
        img_dir='.',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        train_split=config['train_split']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Class names: {class_names}")
    
    # Create model
    print("Creating model...")
    model = get_model(
        model_type=config['model_type'],
        num_classes=config['num_classes'],
        pretrained=True,
        model_name=config['model_name']
    )
    model.to(device)
    
    # Create trainer
    trainer = BeltAlignmentTrainer(model, train_loader, val_loader, device, config)
    
    # Train the model
    trainer.train(config['num_epochs'])
    
    # Save final model
    trainer.save_model('final_model.pth')
    
    # Save configuration
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\nTraining completed!")
    print("Files saved:")
    print("- final_model.pth: Final trained model")
    print("- best_model.pth: Best model during training")
    print("- training_config.json: Training configuration")
    print("- confusion_matrix.png: Confusion matrix plot")
    print("- training_history.png: Training history plot")

if __name__ == "__main__":
    main() 