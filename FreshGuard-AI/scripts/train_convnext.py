"""
FreshGuard AI — Model 2 Training Script: ConvNeXt-Tiny
Trains ConvNeXt-Tiny using two-stage transfer learning with early stopping on validation F1.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dataset import get_dataloaders
from models.convnext_tiny import get_convnext_tiny

def train_convnext(epochs_s1=config.STAGE1_EPOCHS, epochs_s2=config.STAGE2_EPOCHS):
    device = config.DEVICE
    print("\n" + "="*60)
    print(f"  Training ConvNeXt-Tiny | Device: {device}")
    print("="*60)

    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    model = get_convnext_tiny(num_classes=config.NUM_CLASSES, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()

    history = {
        'epoch': [], 'stage': [], 'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [], 'val_f1': []
    }
    best_val_f1 = 0.0
    checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, "convnext_tiny_best.pth")

    # ── STAGE 1: Freeze features backbone, train classifier head ────────────
    print("\n>>> STAGE 1: Training Classifier Head (Backbone Frozen)...")
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.classifier.parameters(), lr=config.STAGE1_LR, weight_decay=config.WEIGHT_DECAY)

    current_epoch = 1
    for epoch in range(epochs_s1):
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []

        for X, y in tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{epochs_s1}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_preds.extend(torch.argmax(out, 1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                val_loss += criterion(out, y).item() * X.size(0)
                val_preds.extend(torch.argmax(out, 1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        print(f"Epoch {current_epoch} (S1) | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        history['epoch'].append(current_epoch)
        history['stage'].append('Stage1')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f" -> Saved new best checkpoint to {checkpoint_path} (Val F1: {best_val_f1:.4f})")

        current_epoch += 1

    # ── STAGE 2: Unfreeze upper backbone, fine-tune ──────────────────────────
    print("\n>>> STAGE 2: End-to-End Fine-Tuning (Unfrozen Backbone)...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': config.STAGE2_LR * 0.1},
        {'params': model.classifier.parameters(), 'lr': config.STAGE2_LR}
    ], weight_decay=config.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_s2)

    for epoch in range(epochs_s2):
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []

        for X, y in tqdm(train_loader, desc=f"Stage 2 Epoch {epoch+1}/{epochs_s2}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_preds.extend(torch.argmax(out, 1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        scheduler.step()

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                val_loss += criterion(out, y).item() * X.size(0)
                val_preds.extend(torch.argmax(out, 1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        print(f"Epoch {current_epoch} (S2) | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        history['epoch'].append(current_epoch)
        history['stage'].append('Stage2')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f" -> Saved new best checkpoint to {checkpoint_path} (Val F1: {best_val_f1:.4f})")

        current_epoch += 1

    # Save history CSV
    df_hist = pd.DataFrame(history)
    hist_csv = os.path.join(config.METRICS_DIR, "convnext_tiny_history.csv")
    df_hist.to_csv(hist_csv, index=False)

    # Save learning curves plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df_hist['epoch'], df_hist['train_loss'], label='Train Loss', marker='o')
    plt.plot(df_hist['epoch'], df_hist['val_loss'], label='Val Loss', marker='s')
    plt.title('ConvNeXt-Tiny Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(df_hist['epoch'], df_hist['train_acc'], label='Train Acc', marker='o')
    plt.plot(df_hist['epoch'], df_hist['val_acc'], label='Val Acc', marker='s')
    plt.plot(df_hist['epoch'], df_hist['val_f1'], label='Val F1', marker='^')
    plt.title('ConvNeXt-Tiny Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(config.PLOTS_DIR, "convnext_tiny_learning_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nTraining complete for ConvNeXt-Tiny. Saved plot to {plot_path}")
    return model, history

if __name__ == "__main__":
    train_convnext()
