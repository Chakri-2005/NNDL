"""
FreshGuard AI — PyTorch Dataset & Dataloader Infrastructure
Reads leakage-safe split CSVs (train.csv, val.csv, test.csv) and applies appropriate transformations.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import config

class FreshnessDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to split CSV file (filepath, target_class, label, group_id, sha256).
            transform (callable, optional): PyTorch transform pipeline.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row['filepath']
        full_path = os.path.join(config.BASE_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path
        
        image = Image.open(full_path).convert('RGB')
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    """
    Returns train and val/test transforms standardized to 224x224.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_test_transforms

def get_dataloaders(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    """
    Creates DataLoaders for train, val, and test splits.
    """
    train_csv = os.path.join(config.SPLITS_DIR, "train.csv")
    val_csv = os.path.join(config.SPLITS_DIR, "val.csv")
    test_csv = os.path.join(config.SPLITS_DIR, "test.csv")

    for path in [train_csv, val_csv, test_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Split CSV not found at {path}. Run scripts/prepare_dataset.py first.")

    train_tf, val_test_tf = get_transforms()

    train_ds = FreshnessDataset(train_csv, transform=train_tf)
    val_ds = FreshnessDataset(val_csv, transform=val_test_tf)
    test_ds = FreshnessDataset(test_csv, transform=val_test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, test_loader, config.CLASS_NAMES

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print(f"Loaded Dataloaders | Classes: {classes}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
