from datasets import load_dataset, Features, Value, Sequence
from config import DS_DIR_BASE, DATASET_NAME, USERNAME, BERT_HF_NAME, SIGLIP_HF_NAME, MODELS_DIR_BASE
from transformers import SiglipTokenizer, SiglipTextModel, AutoTokenizer, ModernBertModel
import time
import datasets
import torch
from dataset.shapebatching_dataset import ShapeBatchingDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    datasets.config.HF_HUB_OFFLINE = 1
    ds = load_dataset(f"{USERNAME}/{DATASET_NAME}", cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}", num_proc=8, split="train")
    ds = ds.to_iterable_dataset(1000)

    device="cuda"

    siglip_model = SiglipTextModel.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip").to(device)
    siglip_tokenizer = SiglipTokenizer.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip")
    bert_model = ModernBertModel.from_pretrained(BERT_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/modernbert").to(device)
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/modernbert")

    ds = ShapeBatchingDataset(ds, 16, siglip_tokenizer, siglip_model, bert_tokenizer, bert_model, 32, device)

    t = time.time()
    for row in ds:
        print(f"Took: {time.time() - t}")

        t = time.time()
