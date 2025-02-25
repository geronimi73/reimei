import random
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import SiglipTextModel, SiglipTokenizer
from config import MODELS_DIR_BASE, SIGLIP_HF_NAME, MAX_CAPTION_LEN

class ImageNet96Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, bs, device, dtype, col_label="label", col_latent="latent"):
        self.hf_ds=hf_ds
        self.col_label, self.col_latent = col_label, col_latent
        self.prompt_len = 50
        self.device = device
        self.dtype = dtype
        
        self.dataloader = DataLoader(
            hf_ds, collate_fn=self.collate, batch_size=bs, num_workers=1, prefetch_factor=2
        )

    def collate(self, items):
        labels = [i[self.col_label] for i in items]
        # [B, 32, 3, 3]
        latents = torch.Tensor([i[self.col_latent] for i in items]).squeeze()
        # # pick random augmentation -> [B, num_aug, 32, 3, 2]
        latents = latents[:, random.randint(0,3)]
        
        return labels, latents
        
    def __iter__(self):
        while True:
            for labels, latents in self.dataloader:
                # emb, vec = self.encode_siglip(labels)
                yield {"caption": labels, "ae_latent": latents}
    
    # @torch.no_grad
    # def encode_siglip(self, captions):
    #     # Encode the captions
    #     s_tokens = self.siglip_tokenizer(captions, padding='longest', truncation=True, return_tensors="pt", max_length=MAX_CAPTION_LEN).to(self.device)
    #     siglip_outputs = self.siglip_model(**s_tokens, output_hidden_states=True)
    #     siglip_embedding = siglip_outputs.hidden_states[-1]
    #     siglip_vec = siglip_outputs.pooler_output

    #     return siglip_embedding, siglip_vec

    @torch.no_grad
    def calculate_score(self, images, captions):
        """
        Computes similarity scores between generated images and captions using SigLIP.
        """
        # Tokenize captions
        text_inputs = self.siglip_processor.tokenizer(
            captions, padding='longest', truncation=True, return_tensors="pt", max_length=MAX_CAPTION_LEN
        ).to(self.device)
        
        # Encode text
        text_embeddings = self.siglip_model.get_text_features(**text_inputs, output_hidden_states=True)
        
        # Encode images
        image_inputs = self.siglip_processor(images=images.to(torch.float32), return_tensors="pt").to(self.device)
        image_outputs = self.siglip_model.get_image_features(**image_inputs)  # [B, D]
        
        # Compute cosine similarity
        scores = torch.nn.functional.cosine_similarity(image_outputs, text_embeddings, dim=-1)
        return scores

    # def __len__(self):
        # return len(self.dataloader)