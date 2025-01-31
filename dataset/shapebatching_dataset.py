import torch
from torch.utils.data import IterableDataset
from collections import defaultdict
import numpy as np
from torch.utils.data import DataLoader

def custom_collate(batch):
    captions = [item['caption'] for item in batch]
    ae_latents = [item['ae_latent'] for item in batch]
    ae_latent_shapes = [item['ae_latent_shape'] for item in batch]

    return {
        'caption': captions,
        'ae_latent': ae_latents,
        'ae_latent_shape': ae_latent_shapes
    }

class ShapeBatchingDataset(IterableDataset):
    def __init__(self, hf_dataset, batch_size, siglip_tokenizer, siglip_model, bert_tokenizer, bert_model, device, num_workers, shuffle=True, seed=42, buffer_multiplier=20, ):
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
                self.dataloader = DataLoader(self.dataset, self.batch_size, num_workers=self.num_workers, collate_fn=custom_collate)
            
            shape_batches = defaultdict(lambda: {'caption': [], 'ae_latent': []})
            for batch in self.dataloader:
                caption = batch['caption']
                ae_latent = batch['ae_latent']
                ae_latent_shape = batch['ae_latent_shape']

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
        ae_latent = torch.tensor(np.stack([np.array(s, dtype=np.float16).copy() for s in samples['ae_latent']])).reshape(-1, *latent_shape)
        # ae_latent = torch.stack([s['ae_latent'].reshape(*ae_latent_shape) for s in samples])

        s_tokens = self.siglip_tokenizer(samples["caption"], padding='longest', truncation=True, return_tensors="pt").to(self.device)
        b_tokens = self.bert_tokenizer(["[CLS]"+ s for s in samples["caption"]], padding='longest', truncation=True, return_tensors="pt", max_length=65).to(self.device)

        siglip_outputs = self.siglip_model(**s_tokens, output_hidden_states=True)
        siglip_embedding = siglip_outputs.hidden_states[-1]
        siglip_vec = siglip_outputs.pooler_output

        bert_outputs = self.bert_model(**b_tokens, output_hidden_states=True).hidden_states[-1] # (bs, 65, 768). The 65 is CLS + 64 tokens. So we need to seperate the CLS token from the rest.
        bert_vec = bert_outputs[:, 0, :] # (bs, 1024)
        bert_embedding = bert_outputs[:, 1:, :] # (bs, 64, 1024)

        batch = {
            'caption': samples["caption"],
            'ae_latent': ae_latent,
            'ae_latent_shape': latent_shape,
            'siglip_emb': siglip_embedding,
            'siglip_vec': siglip_vec,
            'bert_emb': bert_embedding,
            'bert_vec': bert_vec,
        }
        return batch
