import torch
from torch.utils.data import IterableDataset
from collections import defaultdict
import numpy as np
from torch.utils.data import DataLoader
from config import DATASET_NAME, DS_DIR_BASE, MAX_CAPTION_LEN, MODELS_DIR_BASE, SIGLIP_EMBED_DIM, BERT_EMBED_DIM, SIGLIP_HF_NAME, USERNAME
import random
from transformers import SiglipTextModel, SiglipTokenizer
from datasets import load_dataset

# def custom_collate(batch):
#     captions = [item['caption'] for item in batch]
#     ae_latents = [item['ae_latent'] for item in batch]
#     ae_latent_shapes = [item['ae_latent_shape'] for item in batch]

#     return {
#         'caption': captions,
#         'ae_latent': ae_latents,
#         'ae_latent_shape': ae_latent_shapes
#     }

# def custom_collate(batch):
#     from walloc.walloc import pil_to_latent

#     captions = [item['caption'] for item in batch]
#     labels = [item['cls'] for item in batch]
#     ae_latents = [item['latent'] for item in batch]
#     ae_latents = [pil_to_latent([latent], N=36, n_bits=8, C=4)[:, :32].to(torch.int8).view(torch.float8_e4m3fn).to(torch.bfloat16) for latent in ae_latents]

#     ae_latent_shapes = [item.shape for item in ae_latents]

#     return {
#         'caption': captions,
#         'label': labels,
#         'ae_latent': ae_latents,
#         'ae_latent_shape': ae_latent_shapes
#     }

def custom_collate(batch):
    captions = [item['caption'] for item in batch]
    
    ae_latents = [torch.tensor(item['vae_latent'], dtype=torch.uint8).view(torch.float8_e5m2) for item in batch]
    ae_latent_shapes = [item.shape for item in ae_latents]

    return {
        'caption': captions,
        'ae_latent': ae_latents,
        'ae_latent_shape': ae_latent_shapes
    }


class ShapeBatchingDataset(IterableDataset):
    def __init__(self, hf_dataset, batch_size, siglip_tokenizer, siglip_model, bert_tokenizer, bert_model, device, num_workers, shuffle=True, seed=42, buffer_multiplier=20, ):
    # def __init__(self, hf_dataset, batch_size, device, num_workers, shuffle=True, seed=42, buffer_multiplier=20, ):
        self.dataset = hf_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_multiplier = buffer_multiplier
        self.siglip_tokenizer = siglip_tokenizer
        self.siglip_model = siglip_model
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model
        self.device = device
        self.num_workers = num_workers

    def __iter__(self):
        while True:
            if self.shuffle:
                self.dataset = self.dataset.shuffle(seed=self.seed, buffer_size=self.batch_size*self.buffer_multiplier)
            self.dataloader = DataLoader(self.dataset, self.batch_size * 2, prefetch_factor=5, num_workers=self.num_workers, collate_fn=custom_collate)
            
            shape_batches = defaultdict(lambda: {'caption': [], 'ae_latent': []})
            for batch in self.dataloader:
                caption = batch['caption']
                ae_latent = batch['ae_latent']
                ae_latent_shape = batch['ae_latent_shape']
                # label = batch['label']

                for i in range(len(caption)):
                    shape_key = tuple(ae_latent_shape[i])
                    shape_batches[shape_key]['caption'].append(caption[i])
                    shape_batches[shape_key]['ae_latent'].append(ae_latent[i])

                    # If enough samples are accumulated for this shape, yield a batch
                    if len(shape_batches[shape_key]['caption']) == self.batch_size:
                        batch = self.prepare_batch(shape_batches[shape_key], shape_key)
                        yield batch
                        shape_batches[shape_key]['caption'] = []
                        shape_batches[shape_key]['ae_latent'] = []

    def prepare_batch(self, samples, latent_shape):# -> dict[str, Any]:
        # Convert lists of samples into tensors
        # ae_latent = torch.tensor(np.stack([np.frombuffer(s, dtype=np.float32).copy() for s in samples["ae_latent"]])).reshape(-1, *latent_shape)
        # ae_latent = torch.tensor(np.stack([np.array(s, dtype=np.float16).copy() for s in samples['ae_latent']])).reshape(-1, *latent_shape)
        # ae_latent = torch.stack([s['ae_latent'].reshape(*ae_latent_shape) for s in samples])
        ae_latent = torch.stack(samples["ae_latent"]).squeeze(1)

        # if bool(random.getrandbits(1)):
        siglip_embedding, siglip_vec = self.encode_siglip(samples["caption"])
        # else:
        # _, _, bert_embedding, bert_vec = self.encode_bert(samples["caption"])

        # siglip_embedding, siglip_vec, bert_embedding, bert_vec = self.encode(samples["caption"])

        batch = {
            'caption': samples["caption"],
            'ae_latent': ae_latent,
            'ae_latent_shape': latent_shape,
            'siglip_emb': siglip_embedding,
            'siglip_vec': siglip_vec,
        }
        return batch

    # Encodes a batch of strings into a batch of embeddings
    @torch.no_grad
    def encode(self, captions):
        # Encode the captions
        s_tokens = self.siglip_tokenizer(captions, padding='longest', truncation=True, return_tensors="pt", max_length=MAX_CAPTION_LEN).to(self.device)
        b_tokens = self.bert_tokenizer(["[CLS]"+ c for c in captions], padding='longest', truncation=True, return_tensors="pt", max_length=MAX_CAPTION_LEN + 1).to(self.device)

        # Get the embeddings
        siglip_outputs = self.siglip_model(**s_tokens, output_hidden_states=True)
        siglip_embedding = siglip_outputs.hidden_states[-1]
        siglip_vec = siglip_outputs.pooler_output

        bert_outputs = self.bert_model(**b_tokens, output_hidden_states=True).hidden_states[-1] # (bs, 65, 768). The 65 is CLS + 64 tokens. So we need to seperate the CLS token from the rest.
        bert_vec = bert_outputs[:, 0, :] # (bs, 1024)
        bert_embedding = bert_outputs[:, 1:, :] # (bs, 64, 1024)

        return siglip_embedding, siglip_vec, bert_embedding, bert_vec
    
    # Encodes with only siglip, 0s for bert
    @torch.no_grad
    def encode_siglip(self, captions):
        bs = len(captions)

        # Encode the captions
        s_tokens = self.siglip_tokenizer(captions, padding='longest', truncation=True, return_tensors="pt", max_length=MAX_CAPTION_LEN).to(self.device)
        siglip_outputs = self.siglip_model(**s_tokens, output_hidden_states=True)
        siglip_embedding = siglip_outputs.hidden_states[-1]
        siglip_vec = siglip_outputs.pooler_output

        return siglip_embedding, siglip_vec
    
    @torch.no_grad
    def encode_bert(self, captions):
        bs = len(captions)

        # Encode the captions
        b_tokens = self.bert_tokenizer(["[CLS]"+ c for c in captions], padding='longest', truncation=True, return_tensors="pt", max_length=MAX_CAPTION_LEN + 1).to(self.device)

        bert_outputs = self.bert_model(**b_tokens, output_hidden_states=True).hidden_states[-1] # (bs, 65, 768). The 65 is CLS + 64 tokens. So we need to seperate the CLS token from the rest.
        bert_vec = bert_outputs[:, 0, :] # (bs, 1024)
        bert_embedding = bert_outputs[:, 1:, :] # (bs, 64, 1024)

        return bert_embedding, bert_vec
    
def get_dataset(bs, seed, device, dtype, num_workers=16):
    # ds = load_dataset(f"{USERNAME}/{DATASET_NAME}", cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}", split="train", streaming=True)
    ds = load_dataset(f"{USERNAME}/{DATASET_NAME}", cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}", num_proc=num_workers, split="train")
    ds = ds.to_iterable_dataset(1000)
    siglip_model = SiglipTextModel.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip").to(device, dtype)
    siglip_tokenizer = SiglipTokenizer.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip")
    # bert_model = ModernBertModel.from_pretrained(BERT_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/modernbert").to(device, DTYPE)
    # bert_tokenizer = AutoTokenizer.from_pretrained(BERT_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/modernbert")
    
    # ds = ShapeBatchingDataset(ds, bs, device, num_workers, shuffle=True, seed=seed)
    ds = ShapeBatchingDataset(ds, bs, siglip_tokenizer, siglip_model, None, None, device, num_workers, shuffle=True, seed=seed)
    return ds
