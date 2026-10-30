import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_crop_data_loaders(crop_name="beans", batch_size=32):
    """
    Returns train and validation DataLoaders for the specified crop 
    ('beans', 'maize', or 'cassava') mapped via ImageFolder.
    """
    base_dirs = {
        "beans": "./dataset_beans",
        "maize": "./dataset_maize",
        "cassava": "./dataset_cassava"
    }
    
    data_dir = base_dirs.get(crop_name.lower())
    if not data_dir or not os.path.exists(data_dir):
        raise ValueError(f"Dataset directory for '{crop_name}' does not exist. Run download_datasets.py first.")

    # Field-simulation augmentations for training data
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Standard preprocessing for validation data
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Loaded [{crop_name.upper()}]: {len(train_dataset)} training images, {len(val_dataset)} validation images.")
    print(f"Detected Classes: {train_dataset.classes}")
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test loading loaders for beans
    # train_loader, val_loader = get_crop_data_loaders("beans")
    print("Dataset loader script initialized successfully.")