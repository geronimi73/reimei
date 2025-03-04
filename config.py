USERNAME = "SwayStar123"
# DATASET_NAME = "FFHQ_1024_DC-AE_f32"
# DATASET_NAME = "pruned_preprocessed_commoncatalog-cc-by_DCAE"
# DATASET_NAME = "preprocessed_commoncatalog-cc-by"
# DATASET_NAME = "imagenet_288_dcae_fp8_captions"
# DATASET_NAME = "g-ronimo/IN1k96-augmented-latents_dc-ae-f32c32-sana-1.0"
DATASET_NAME = "imagenet1k_eqsdxlvae_latents"
DS_DIR_BASE = "../../datasets"
MODELS_DIR_BASE = "../../models"
# AE_SCALING_FACTOR = 0.13025 # sdxl-vae-fp16-fix
# AE_SCALING_FACTOR = 0.3189 # f32c32-in-1.0
# AE_SCALING_FACTOR = 0.41407 # f32-c32-sana-1.1
# AE_SCALING_FACTOR = 0.4552 # f32-c32-mix-1.0
AE_SCALING_FACTOR = 0.12746273743957862 # eq-sdxl
AE_SHIFT_FACTOR = 0.8640247167934477 # eq-sdxl

BS = 128
TRAIN_STEPS = 300_000
MASK_RATIO = 0.75 # Percent to mask
CFG_RATIO = 0.1 # Percent to drop
MAX_CAPTION_LEN = 32 # Token length to encode

LR = 0.0001

# AE_HF_NAME = "madebyollin/sdxl-vae-fp16-fix"
# AE_HF_NAME = "dc-ae-f32c32-in-1.0-diffusers"
# AE_HF_NAME = "dc-ae-f32c32-sana-1.1-diffusers"
# AE_HF_NAME = "dc-ae-f32c32-mix-1.0-diffusers"
# AE_HF_NAME = "dc-ae-f32c32-sana-1.0-diffusers"
AE_HF_NAME = "KBlueLeaf/EQ-SDXL-VAE"

if "dc-ae" in AE_HF_NAME:
    AE_CHANNELS = 32
else:
    AE_CHANNELS = 4

SIGLIP_HF_NAME = "google/siglip-so400m-patch14-384"
SIGLIP_EMBED_DIM = 1152
BERT_HF_NAME = "answerdotai/ModernBERT-large"
BERT_EMBED_DIM = 1024

SEED = 42

DIT_G = dict(
    num_layers=40,
    num_heads=16,
    embed_dim=1408,
)
DIT_XL = dict(
    num_layers=28,
    num_heads=16,
    embed_dim=1152,
)
DIT_L = dict(
    num_layers=24,
    num_heads=16,
    embed_dim=1024,
)
DIT_B = dict(
    num_layers=12,
    num_heads=12,
    embed_dim=768,
)
DIT_S = dict(
    num_layers=12,
    num_heads=6,
    embed_dim=384,
)
