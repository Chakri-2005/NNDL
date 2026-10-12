"""
MobileViT-Dedicated Conditional Fruit GAN
=========================================
Paired with: MobileViT-S classifier.
Architecture: Lightweight separable convolutional GAN featuring self-attention
modulation at the latent bottleneck to simulate global multi-head attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

GAN_ID = "mobilevit_gan"
TARGET_CLASSIFIER = "mobilevit_s"
LATENT_DIM = 80
IMG_SIZE = 64
NUM_CLASSES = 6

class SelfAttentionModulation(nn.Module):
    """Lightweight spatial self-attention block for transformer-like global context."""
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 4, 1)
        self.key   = nn.Conv2d(in_dim, in_dim // 4, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = F.softmax(torch.bmm(q, k) / ((C // 4) ** 0.5), dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out

class MobileViTFruitGenerator(nn.Module):
    """Transformer-augmented lightweight generator for MobileViT-S pairing."""
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.gan_id = GAN_ID
        self.target_classifier = TARGET_CLASSIFIER

        self.label_embedding = nn.Embedding(num_classes, 24)
        input_dim = latent_dim + 24
        self.init_size = IMG_SIZE // 16  # 4x4

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 192 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.attn = SelfAttentionModulation(192)

        self.upsample_blocks = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.Upsample(scale_factor=2),  # 8x8
            nn.Conv2d(192, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # 16x16
            nn.Conv2d(96, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # 32x32
            nn.Conv2d(48, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # 64x64
            nn.Conv2d(24, channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels)
        gen_in = torch.cat((noise, label_emb), dim=-1)
        out = self.fc(gen_in)
        out = out.view(out.shape[0], 192, self.init_size, self.init_size)
        out = self.attn(out)
        return self.upsample_blocks(out)

class MobileViTFruitDiscriminator(nn.Module):
    """Separable convolutional discriminator with spatial attention."""
    def __init__(self, num_classes=NUM_CLASSES, channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.gan_id = GAN_ID
        self.target_classifier = TARGET_CLASSIFIER

        self.label_embedding = nn.Embedding(num_classes, IMG_SIZE * IMG_SIZE)

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + 1, 24, 3, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, stride=2, padding=1),            # 16x16
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.attn = SelfAttentionModulation(48)
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 96, 3, stride=2, padding=1),            # 8x8
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 192, 3, stride=2, padding=1),           # 4x4
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_emb = self.label_embedding(labels).view(labels.shape[0], 1, IMG_SIZE, IMG_SIZE)
        d_in = torch.cat((img, label_emb), dim=1)
        out = self.conv1(d_in)
        out = self.conv2(out)
        out = self.attn(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return self.head(out)
