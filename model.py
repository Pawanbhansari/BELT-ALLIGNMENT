import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class BeltAlignmentClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, model_name='resnet18'):
        super(BeltAlignmentClassifier, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained model
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
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the classifier head"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x):
        """Extract features without classification"""
        return self.backbone(x)

class BeltAlignmentRegressor(nn.Module):
    """Alternative model for regression instead of classification"""
    def __init__(self, pretrained=True, model_name='resnet18'):
        super(BeltAlignmentRegressor, self).__init__()
        self.model_name = model_name
        
        # Load pretrained model
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.regressor(features)
        return output.squeeze()

class AttentionModule(nn.Module):
    """Attention module to focus on belt regions"""
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class BeltAlignmentClassifierWithAttention(nn.Module):
    """Enhanced model with attention mechanism"""
    def __init__(self, num_classes=4, pretrained=True):
        super(BeltAlignmentClassifierWithAttention, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the last few layers
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Add attention module
        self.attention = AttentionModule(512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        attended_features = self.attention(features)
        pooled_features = self.global_pool(attended_features).squeeze()
        output = self.classifier(pooled_features)
        return output

def get_model(model_type='classifier', num_classes=4, pretrained=True, model_name='resnet18'):
    """Factory function to get different model types"""
    if model_type == 'classifier':
        return BeltAlignmentClassifier(num_classes, pretrained, model_name)
    elif model_type == 'regressor':
        return BeltAlignmentRegressor(pretrained, model_name)
    elif model_type == 'attention':
        return BeltAlignmentClassifierWithAttention(num_classes, pretrained)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test classifier
    classifier = get_model('classifier', num_classes=4)
    classifier.to(device)
    print(f"Classifier parameters: {count_parameters(classifier):,}")
    
    # Test regressor
    regressor = get_model('regressor')
    regressor.to(device)
    print(f"Regressor parameters: {count_parameters(regressor):,}")
    
    # Test attention model
    attention_model = get_model('attention', num_classes=4)
    attention_model.to(device)
    print(f"Attention model parameters: {count_parameters(attention_model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        classifier_output = classifier(dummy_input)
        regressor_output = regressor(dummy_input)
        attention_output = attention_model(dummy_input)
        
        print(f"Classifier output shape: {classifier_output.shape}")
        print(f"Regressor output shape: {regressor_output.shape}")
        print(f"Attention output shape: {attention_output.shape}") 