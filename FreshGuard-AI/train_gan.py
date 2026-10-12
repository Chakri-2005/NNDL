"""
FreshGuard AI -- Multi-GAN Independent Training Pipeline
=========================================================
Trains three independent, dedicated fruit GANs:
  1. EfficientNetV2-S Dedicated GAN (Residual DCGAN, Adam lr=2e-4)
  2. ConvNeXt-Tiny Dedicated GAN     (Inverted Bottleneck 7x7 Depthwise, AdamW lr=1.8e-4)
  3. MobileViT-S Dedicated GAN       (Separable Conv + Self-Attention Bottleneck, Adam lr=2.5e-4)

Each GAN maintains:
  - Independent architecture & parameter weights
  - Dedicated checkpoint path: results/checkpoints/gans/<gan_name>_best.pth
  - Dedicated training logs, loss curves, and synthetic sample gallery
  - Dedicated random seed and optimizer state

Usage:
  python train_gan.py --model efficientnetv2_s --epochs 5
  python train_gan.py --model convnext_tiny --epochs 5
  python train_gan.py --model mobilevit_s --epochs 5
  python train_gan.py --all --epochs 5
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image

# Repository imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset import CLASS_NAMES, NUM_CLASSES, FreshnessDataset
from models.gans import GAN_REGISTRY, resolve_gan_key, create_gan

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 64
CKPT_DIR = os.path.join(config.RESULTS_DIR, "checkpoints", "gans")
GAN_RESULTS_DIR = os.path.join(config.RESULTS_DIR, "testing", "gan")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(GAN_RESULTS_DIR, exist_ok=True)

def train_single_gan(
    model_key: str,
    epochs: int = 5,
    batch_size: int = 32,
    seed: int = 42,
    save_samples: bool = True
) -> Dict:
    resolved_key = resolve_gan_key(model_key)
    cfg = GAN_REGISTRY[resolved_key]
    display_name = cfg['display_name']
    ckpt_path = os.path.join(CKPT_DIR, cfg['checkpoint_file'])
    out_dir = os.path.join(GAN_RESULTS_DIR, cfg['output_subdir'])
    samples_dir = os.path.join(out_dir, "synthetic_samples")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  Training {display_name} | Key: {resolved_key}")
    print(f"  Target Classifier : {cfg['classifier_key']}")
    print(f"  Latent Dim        : {cfg['latent_dim']} | Optimizer: {cfg['opt_type']} (lr={cfg['lr']})")
    print(f"  Checkpoint Output : {ckpt_path}")
    print(f"  Device            : {DEVICE}")
    print("=" * 70)

    # Set independent deterministic seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate dedicated Generator & Discriminator
    generator = cfg['generator_cls'](latent_dim=cfg['latent_dim']).to(DEVICE)
    discriminator = cfg['discriminator_cls']().to(DEVICE)

    # Optimizer
    if cfg['opt_type'] == 'adamw':
        opt_G = torch.optim.AdamW(generator.parameters(), lr=cfg['lr'], betas=(0.5, 0.999), weight_decay=1e-4)
        opt_D = torch.optim.AdamW(discriminator.parameters(), lr=cfg['lr'], betas=(0.5, 0.999), weight_decay=1e-4)
    else:
        opt_G = torch.optim.Adam(generator.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(discriminator.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    # Data Loader (Real training split only, normalized to [-1, 1])
    gan_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_csv = os.path.join(config.SPLITS_DIR, "train.csv")
    dataset = FreshnessDataset(train_csv, transform=gan_tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    history = {'g_loss': [], 'd_loss': []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        g_losses, d_losses = [], []
        for real_imgs, labels in loader:
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            valid = torch.ones(b_size, 1, device=DEVICE)
            fake = torch.zeros(b_size, 1, device=DEVICE)

            # Train Generator
            opt_G.zero_grad()
            z = torch.randn(b_size, cfg['latent_dim'], device=DEVICE)
            gen_labels = torch.randint(0, NUM_CLASSES, (b_size,), device=DEVICE)
            gen_imgs = generator(z, gen_labels)

            g_loss = criterion(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            opt_G.step()

            # Train Discriminator
            opt_D.zero_grad()
            d_real = criterion(discriminator(real_imgs, labels), valid)
            d_fake = criterion(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (d_real + d_fake) / 2.0
            d_loss.backward()
            opt_D.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        avg_g = float(np.mean(g_losses))
        avg_d = float(np.mean(d_losses))
        history['g_loss'].append(avg_g)
        history['d_loss'].append(avg_d)
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss D: {avg_d:.4f} | Loss G: {avg_g:.4f}")

    train_duration = round(time.time() - start_time, 2)
    print(f"  [OK] Training completed in {train_duration}s.")

    # Save dedicated checkpoint
    torch.save({
        'gan_id': cfg['gan_id'],
        'classifier_key': cfg['classifier_key'],
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'opt_D_state_dict': opt_D.state_dict(),
        'history': history,
        'epochs': epochs,
        'train_duration': train_duration,
        'latent_dim': cfg['latent_dim'],
        'training_seed': seed
    }, ckpt_path)
    print(f"  [OK] Saved dedicated checkpoint -> {ckpt_path}")

    # Generate sample grid (6 classes x 8 samples = 48 samples)
    generator.eval()
    samples_per_class = 8
    num_total = NUM_CLASSES * samples_per_class
    eval_z = torch.randn(num_total, cfg['latent_dim'], device=DEVICE)
    eval_labels = torch.tensor([c for c in range(NUM_CLASSES) for _ in range(samples_per_class)], device=DEVICE)

    with torch.no_grad():
        fake_imgs = generator(eval_z, eval_labels)
    fake_norm = torch.clamp((fake_imgs + 1.0) / 2.0, 0.0, 1.0)

    grid_path = os.path.join(out_dir, "synthetic_samples_grid.png")
    save_image(make_grid(fake_norm, nrow=samples_per_class, padding=2), grid_path)
    print(f"  [OK] Saved sample grid -> {grid_path}")

    # Save individual class sample images for gallery
    if save_samples:
        for c in range(NUM_CLASSES):
            cls_name = CLASS_NAMES[c]
            c_img = fake_norm[c * samples_per_class]  # First sample of class
            c_path = os.path.join(samples_dir, f"{cls_name}_synthetic.png")
            save_image(c_img, c_path)
        print(f"  [OK] Saved individual gallery samples -> {samples_dir}")

    # Plot loss curve
    loss_path = os.path.join(out_dir, "training_loss_curves.png")
    plt.figure(figsize=(8, 4))
    plt.plot(history['g_loss'], label='Generator Loss', color='#10b981', lw=2)
    plt.plot(history['d_loss'], label='Discriminator Loss', color='#ef4444', lw=2)
    plt.title(f"{display_name} - Training Dynamics", fontsize=12, weight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()
    print(f"  [OK] Saved loss curves -> {loss_path}")

    return {
        'model_key': resolved_key,
        'gan_id': cfg['gan_id'],
        'checkpoint_path': ckpt_path,
        'epochs': epochs,
        'train_duration': train_duration,
        'final_g_loss': history['g_loss'][-1],
        'final_d_loss': history['d_loss'][-1]
    }

def main():
    parser = argparse.ArgumentParser(description="Train FreshGuard AI Dedicated Fruit GANs")
    parser.add_argument('--model', choices=['efficientnetv2_s', 'convnext_tiny', 'mobilevit_s', 'all'],
                        default='all', help="Specific GAN model to train or 'all'")
    parser.add_argument('--all', action='store_true', help="Train all 3 GANs sequentially")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs per GAN")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    args = parser.parse_args()

    targets = list(GAN_REGISTRY.keys()) if (args.all or args.model == 'all') else [args.model]

    print("\n" + "=" * 70)
    print("  FreshGuard AI: Multi-GAN Independent Training Pipeline")
    print(f"  Targets: {targets}")
    print("=" * 70)

    summary = []
    for idx, key in enumerate(targets):
        seed = 42 + idx * 10
        res = train_single_gan(key, epochs=args.epochs, batch_size=args.batch_size, seed=seed)
        summary.append(res)

    print("\n" + "=" * 70)
    print("  All Target GANs Trained Successfully!")
    for s in summary:
        print(f"   - {s['gan_id']}: G_Loss={s['final_g_loss']:.4f}, D_Loss={s['final_d_loss']:.4f} ({s['train_duration']}s)")
    print("=" * 70)

if __name__ == '__main__':
    main()
