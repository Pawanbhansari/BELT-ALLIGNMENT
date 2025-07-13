import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class BeltSegmentationUNet(nn.Module):
    """
    U-Net based segmentation model for belt alignment analysis
    This approach provides pixel-level classification of belt regions
    """
    
    def __init__(self, num_classes=4, pretrained=True):
        super(BeltSegmentationUNet, self).__init__()
        self.num_classes = num_classes
        
        # Encoder (ResNet18 backbone)
        resnet = models.resnet18(pretrained=pretrained)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 channels
        self.encoder2 = resnet.layer1  # 64 channels
        self.encoder3 = resnet.layer2  # 128 channels
        self.encoder4 = resnet.layer3  # 256 channels
        self.encoder5 = resnet.layer4  # 512 channels
        
        # Decoder (U-Net style upsampling)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),  # 96 = 64 + 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final segmentation head
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)      # 224x224 -> 112x112
        e2 = self.encoder2(e1)     # 112x112 -> 56x56
        e3 = self.encoder3(e2)     # 56x56 -> 28x28
        e4 = self.encoder4(e3)     # 28x28 -> 14x14
        e5 = self.encoder5(e4)     # 14x14 -> 7x7
        
        # Decoder path with skip connections
        d4 = self.up4(e5)          # 7x7 -> 14x14
        d4 = torch.cat([d4, e4], dim=1)  # Skip connection
        d4 = self.conv4(d4)
        
        d3 = self.up3(d4)          # 14x14 -> 28x28
        d3 = torch.cat([d3, e3], dim=1)  # Skip connection
        d3 = self.conv3(d3)
        
        d2 = self.up2(d3)          # 28x28 -> 56x56
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        d2 = self.conv2(d2)
        
        d1 = self.up1(d2)          # 56x56 -> 112x112
        # Ensure e1 is the right size for skip connection
        if e1.size(2) != d1.size(2) or e1.size(3) != d1.size(3):
            e1 = F.interpolate(e1, size=d1.size()[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d1 = self.conv1(d1)
        
        # Final output
        output = self.final(d1)    # 112x112 -> 112x112x4
        
        # Upsample to original size
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return output

class BeltAlignmentSegmentationTrainer:
    """
    Trainer for segmentation-based belt alignment analysis
    """
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        # Loss function for segmentation
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate IoU (Intersection over Union)
                pred_masks = torch.argmax(outputs, dim=1)
                iou = self.calculate_iou(pred_masks, masks)
                
                total_loss += loss.item()
                total_iou += iou
        
        return total_loss / len(val_loader), total_iou / len(val_loader)
    
    def calculate_iou(self, pred, target):
        """Calculate IoU for segmentation"""
        intersection = (pred == target).sum()
        union = pred.numel()
        return intersection.float() / union

def create_segmentation_dataset(csv_file, img_dir, mask_dir=None):
    """
    Create segmentation dataset
    For this example, we'll create synthetic masks based on center_percent
    """
    import pandas as pd
    from PIL import Image
    import numpy as np
    
    df = pd.read_csv(csv_file)
    
    class SegmentationDataset(torch.utils.data.Dataset):
        def __init__(self, df, img_dir, transform=None):
            self.df = df
            self.img_dir = img_dir
            self.transform = transform
            
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row['filename'])
            image = Image.open(img_path).convert('RGB')
            
            # Create synthetic mask based on center_percent
            mask = self.create_synthetic_mask(row['center_percent'], image.size)
            
            if self.transform:
                image = self.transform(image)
                mask = torch.from_numpy(mask).long()
            
            return image, mask
        
        def create_synthetic_mask(self, center_percent, img_size):
            """Create synthetic segmentation mask"""
            width, height = img_size
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Create belt region (horizontal strip in middle)
            belt_height = height // 4
            belt_start = (height - belt_height) // 2
            
            # Determine belt alignment class
            if center_percent < -50:
                class_id = 0  # Severe Left
            elif center_percent < 0:
                class_id = 1  # Mild Left
            elif center_percent <= 100:
                class_id = 2  # Good
            else:
                class_id = 3  # Severe Right
            
            # Create belt mask with offset based on center_percent
            offset = int((center_percent / 100) * width * 0.3)  # 30% of width
            belt_center = width // 2 + offset
            
            # Draw belt region
            belt_width = width // 3
            belt_left = max(0, belt_center - belt_width // 2)
            belt_right = min(width, belt_center + belt_width // 2)
            
            if belt_right > belt_left:
                belt_region = np.ones((belt_height, belt_right - belt_left), dtype=np.uint8)
                # Add some noise to make it more realistic
                noise = np.random.random(belt_region.shape) > 0.1
                belt_region = belt_region * noise
                mask[belt_start:belt_start+belt_height, belt_left:belt_right] = belt_region
            
            return mask
    
    return SegmentationDataset(df, img_dir)

def visualize_segmentation_results(model, image_path, device):
    """
    Visualize segmentation results
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Color-coded segmentation mask
    colors = ['red', 'orange', 'green', 'blue']
    colored_mask = np.zeros((*pred_mask.shape, 3))
    for i, color in enumerate(colors):
        colored_mask[pred_mask == i] = plt.cm.colors.to_rgb(color)
    
    ax2.imshow(colored_mask)
    ax2.set_title('Segmentation Mask\n(Red: Severe Left, Orange: Mild Left, Green: Good, Blue: Severe Right)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create model
    model = BeltSegmentationUNet(num_classes=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("Segmentation Model Architecture:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output classes per pixel: {output.shape[1]}") 