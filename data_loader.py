import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class BeltAlignmentDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, classification_mode=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            img_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
            classification_mode (bool): If True, convert regression to classification
        """
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classification_mode = classification_mode
        
        if classification_mode:
            # Convert regression values to classification labels
            self.labels_df['alignment_class'] = self._convert_to_classes(self.labels_df['center_percent'])
    
    def _convert_to_classes(self, center_percent):
        """Convert center_percent to classification labels"""
        classes = []
        for value in center_percent:
            if value < -50:
                classes.append(0)  # Severe misalignment (left)
            elif value < 0:
                classes.append(1)  # Mild misalignment (left)
            elif value <= 100:
                classes.append(2)  # Good alignment
            else:
                classes.append(3)  # Severe misalignment (right)
        return classes
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        if self.classification_mode:
            label = self.labels_df.iloc[idx]['alignment_class']
        else:
            label = self.labels_df.iloc[idx]['center_percent']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self):
        """Get class names for classification"""
        return ['Severe Left Misalignment', 'Mild Left Misalignment', 'Good Alignment', 'Severe Right Misalignment']

def get_transforms(img_size=224):
    """Get training and validation transforms"""
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

def create_data_loaders(csv_file, img_dir, batch_size=4, img_size=224, train_split=0.8):
    """Create train and validation data loaders"""
    # Load dataset
    dataset = BeltAlignmentDataset(csv_file, img_dir, transform=None, classification_mode=True)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Apply transforms
    train_transform, val_transform = get_transforms(img_size)
    
    # Create datasets with transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, dataset.get_class_names()

def visualize_dataset(dataset, num_samples=4):
    """Visualize samples from the dataset"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    class_names = dataset.get_class_names()
    
    for i in range(num_samples):
        img, label = dataset[i]
        
        # Denormalize image
        img = img.numpy().transpose((1, 2, 0))
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'Class: {class_names[label]} ({label})')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test the data loader
    dataset = BeltAlignmentDataset('labels.csv', '.', transform=get_transforms()[1])
    print(f"Dataset size: {len(dataset)}")
    print(f"Class names: {dataset.get_class_names()}")
    
    # Visualize a few samples
    visualize_dataset(dataset) 