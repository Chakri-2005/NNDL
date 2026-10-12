"""
FreshGuard AI -- Multi-GAN Registry & Model Loader
==================================================
Manages three independent, dedicated Generative Adversarial Networks:
  1. efficientnet_gan -> Dedicated to EfficientNetV2-S
  2. convnext_gan     -> Dedicated to ConvNeXt-Tiny
  3. mobilevit_gan    -> Dedicated to MobileViT-S
"""
from models.gans.efficientnet_gan.model import (
    EfficientNetFruitGenerator, EfficientNetFruitDiscriminator,
    LATENT_DIM as EFF_LATENT_DIM
)
from models.gans.convnext_gan.model import (
    ConvNeXtFruitGenerator, ConvNeXtFruitDiscriminator,
    LATENT_DIM as CONV_LATENT_DIM
)
from models.gans.mobilevit_gan.model import (
    MobileViTFruitGenerator, MobileViTFruitDiscriminator,
    LATENT_DIM as MOB_LATENT_DIM
)

GAN_REGISTRY = {
    'efficientnetv2_s': {
        'gan_id': 'efficientnet_gan',
        'classifier_key': 'efficientnetv2_s',
        'display_name': 'EfficientNetV2-S Dedicated GAN',
        'generator_cls': EfficientNetFruitGenerator,
        'discriminator_cls': EfficientNetFruitDiscriminator,
        'latent_dim': EFF_LATENT_DIM,
        'lr': 2e-4,
        'opt_type': 'adam',
        'checkpoint_file': 'efficientnet_gan_best.pth',
        'output_subdir': 'efficientnetv2_s'
    },
    'convnext_tiny': {
        'gan_id': 'convnext_gan',
        'classifier_key': 'convnext_tiny',
        'display_name': 'ConvNeXt-Tiny Dedicated GAN',
        'generator_cls': ConvNeXtFruitGenerator,
        'discriminator_cls': ConvNeXtFruitDiscriminator,
        'latent_dim': CONV_LATENT_DIM,
        'lr': 1.8e-4,
        'opt_type': 'adamw',
        'checkpoint_file': 'convnext_gan_best.pth',
        'output_subdir': 'convnext_tiny'
    },
    'mobilevit_s': {
        'gan_id': 'mobilevit_gan',
        'classifier_key': 'mobilevit_s',
        'display_name': 'MobileViT-S Dedicated GAN',
        'generator_cls': MobileViTFruitGenerator,
        'discriminator_cls': MobileViTFruitDiscriminator,
        'latent_dim': MOB_LATENT_DIM,
        'lr': 2.5e-4,
        'opt_type': 'adam',
        'checkpoint_file': 'mobilevit_gan_best.pth',
        'output_subdir': 'mobilevit_s'
    }
}

# Alias mapping for flexibility
GAN_ALIASES = {
    'efficientnet': 'efficientnetv2_s',
    'efficientnet_gan': 'efficientnetv2_s',
    'convnext': 'convnext_tiny',
    'convnext_gan': 'convnext_tiny',
    'mobilevit': 'mobilevit_s',
    'mobilevit_gan': 'mobilevit_s'
}

def resolve_gan_key(key: str) -> str:
    k = key.lower().strip()
    return GAN_ALIASES.get(k, k)

def get_gan_config(model_key: str):
    k = resolve_gan_key(model_key)
    if k not in GAN_REGISTRY:
        raise ValueError(f"Unknown GAN key: '{model_key}'. Available: {list(GAN_REGISTRY.keys())}")
    return GAN_REGISTRY[k]

def create_gan(model_key: str):
    cfg = get_gan_config(model_key)
    gen = cfg['generator_cls'](latent_dim=cfg['latent_dim'])
    disc = cfg['discriminator_cls']()
    return gen, disc, cfg
