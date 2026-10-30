import torch
import torch.nn as nn
from torchvision import models

def get_resnet50_teacher(num_classes=3, device='cuda'):
    """
    Initializes a ResNet-50 teacher model with ImageNet weights 
    and custom classification head for crop disease classification.
    """
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    # Freeze backbone layers initially for transfer learning phase
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classification head to match disease classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model.to(device)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Example: 3 classes for beans (healthy, angular leaf spot, bean rust)
    teacher_model = get_resnet50_teacher(num_classes=3, device=device)
    print("ResNet-50 Teacher Model successfully initialized for multi-crop execution!")