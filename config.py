USERNAME = "SwayStar123"
DATASET_NAME = "preprocessed_commoncatalog-cc-by_DCAE"
DS_DIR_BASE = "../../datasets"
MODELS_DIR_BASE = "../../models"
AE_SCALING_FACTOR = 0.3189

BS = 32
EPOCHS = 50
MASK_RATIO = 0.75
CFG_RATIO = 0.02
SEED = 42

LR = 5e-4

AE_HF_NAME = "dc-ae-f32c32-mix-1.0-diffusers"
AE_CHANNELS = 32
SIGLIP_HF_NAME = "google/siglip-so400m-patch14-384"
SIGLIP_EMBED_DIM = 1152
BERT_HF_NAME = "answerdotai/ModernBERT-large"
BERT_EMBED_DIM = 1024

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
