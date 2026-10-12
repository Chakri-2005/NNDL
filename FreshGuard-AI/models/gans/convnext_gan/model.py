"""
ConvNeXt-Dedicated Conditional Fruit GAN
========================================
Paired with: ConvNeXt-Tiny classifier.
Architecture: Deep Inverted-Bottleneck GAN utilizing 7x7 depthwise-separable convs,
GroupNorm, and GELU activations inspired by ConvNeXt design principles.
"""
import torch
import torch.nn as nn

GAN_ID = "convnext_gan"
TARGET_CLASSIFIER = "convnext_tiny"
LATENT_DIM = 128
IMG_SIZE = 64
NUM_CLASSES = 6

class ConvNeXtFruitGenerator(nn.Module):
    """Inverted-bottleneck Conditional Generator for ConvNeXt-Tiny pairing."""
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.gan_id = GAN_ID
        self.target_classifier = TARGET_CLASSIFIER

        self.label_embedding = nn.Embedding(num_classes, 32)
        input_dim = latent_dim + 32
        self.init_size = IMG_SIZE // 16  # 4x4

        self.project = nn.Sequential(
            nn.Linear(input_dim, 384 * self.init_size * self.init_size),
            nn.GELU()
        )

        # Inverted bottleneck upscale blocks with 7x7 depthwise convs
        self.stage1 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 8x8
            nn.Conv2d(384, 384, 7, padding=3, groups=384),
            nn.GroupNorm(8, 384),
            nn.Conv2d(384, 192, 1),
            nn.GELU()
        )
        self.stage2 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 16x16
            nn.Conv2d(192, 192, 7, padding=3, groups=192),
            nn.GroupNorm(8, 192),
            nn.Conv2d(192, 96, 1),
            nn.GELU()
        )
        self.stage3 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 32x32
            nn.Conv2d(96, 96, 7, padding=3, groups=96),
            nn.GroupNorm(8, 96),
            nn.Conv2d(96, 48, 1),
            nn.GELU()
        )
        self.stage4 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 64x64
            nn.Conv2d(48, 48, 7, padding=3, groups=48),
            nn.GroupNorm(6, 48),
            nn.Conv2d(48, channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels)
        gen_in = torch.cat((noise, label_emb), dim=-1)
        out = self.project(gen_in)
        out = out.view(out.shape[0], 384, self.init_size, self.init_size)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return self.stage4(out)

class ConvNeXtFruitDiscriminator(nn.Module):
    """Depthwise-conv discriminator with GroupNorm and GELU."""
    def __init__(self, num_classes=NUM_CLASSES, channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.gan_id = GAN_ID
        self.target_classifier = TARGET_CLASSIFIER

        self.label_embedding = nn.Embedding(num_classes, IMG_SIZE * IMG_SIZE)

        self.stem = nn.Sequential(
            nn.Conv2d(channels + 1, 48, 4, stride=2, padding=1),  # 32x32
            nn.GELU()
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(48, 48, 7, padding=3, groups=48),
            nn.GroupNorm(6, 48),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),             # 16x16
            nn.GELU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 96, 7, padding=3, groups=96),
            nn.GroupNorm(8, 96),
            nn.Conv2d(96, 192, 4, stride=2, padding=1),            # 8x8
            nn.GELU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 192, 7, padding=3, groups=192),
            nn.GroupNorm(8, 192),
            nn.Conv2d(192, 384, 4, stride=2, padding=1),           # 4x4
            nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(384, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_emb = self.label_embedding(labels).view(labels.shape[0], 1, IMG_SIZE, IMG_SIZE)
        d_in = torch.cat((img, label_emb), dim=1)
        out = self.stem(d_in)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        return self.classifier(out)
