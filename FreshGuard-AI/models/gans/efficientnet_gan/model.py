"""
EfficientNet-Dedicated Conditional Fruit GAN
============================================
Paired with: EfficientNetV2-S classifier.
Architecture: Residual Convolutional Conditional DCGAN with fused feature blocks.
"""
import torch
import torch.nn as nn

GAN_ID = "efficientnet_gan"
TARGET_CLASSIFIER = "efficientnetv2_s"
LATENT_DIM = 100
IMG_SIZE = 64
NUM_CLASSES = 6

class EfficientNetFruitGenerator(nn.Module):
    """Residual Conditional DCGAN Generator for EfficientNetV2-S pairing."""
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.gan_id = GAN_ID
        self.target_classifier = TARGET_CLASSIFIER

        self.label_embedding = nn.Embedding(num_classes, num_classes)
        input_dim = latent_dim + num_classes
        self.init_size = IMG_SIZE // 16  # 4x4

        self.l1 = nn.Sequential(nn.Linear(input_dim, 256 * self.init_size * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),  # 8x8
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # 16x16
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # 32x32
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # 64x64
            nn.Conv2d(32, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels)
        gen_input = torch.cat((noise, label_emb), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        return self.conv_blocks(out)

class EfficientNetFruitDiscriminator(nn.Module):
    """Conditional Discriminator with strided convolutions and dropout."""
    def __init__(self, num_classes=NUM_CLASSES, channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.gan_id = GAN_ID
        self.target_classifier = TARGET_CLASSIFIER

        self.label_embedding = nn.Embedding(num_classes, IMG_SIZE * IMG_SIZE)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels + 1, 32, bn=False),  # 32x32
            *discriminator_block(32, 64),                     # 16x16
            *discriminator_block(64, 128),                    # 8x8
            *discriminator_block(128, 256),                   # 4x4
        )
        self.adv_layer = nn.Sequential(nn.Linear(256 * 4 * 4, 1), nn.Sigmoid())

    def forward(self, img, labels):
        label_emb = self.label_embedding(labels).view(labels.shape[0], 1, IMG_SIZE, IMG_SIZE)
        d_in = torch.cat((img, label_emb), dim=1)
        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        return self.adv_layer(out)
