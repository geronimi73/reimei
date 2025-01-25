import torch
from torch.utils.data import IterableDataset
from collections import defaultdict
import numpy as np

class ShapeBatchingDataset(IterableDataset):
    def __init__(self, hf_dataset, batch_size, shuffle=True, seed=42, buffer_multiplier=20):
        self.dataset = hf_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_multiplier = buffer_multiplier

    def __iter__(self):
        if self.shuffle:
            self.dataset = self.dataset.shuffle(seed=self.seed, buffer_size=self.batch_size*self.buffer_multiplier)
        
        shape_batches = defaultdict(list)
        for sample in self.dataset:
            # Get the shape as a tuple to use as a key
            shape_key = tuple(sample['ae_latent_shape'])
            shape_batches[shape_key].append(sample) 

            # If enough samples are accumulated for this shape, yield a batch
            if len(shape_batches[shape_key]) == self.batch_size:
                batch = self.prepare_batch(shape_batches[shape_key])
                yield batch
                shape_batches[shape_key] = []  # Reset the buffer for this shape

        # After iterating over the dataset, yield any remaining partial batches
        for remaining_samples in shape_batches.values():
            if remaining_samples:
                batch = self.prepare_batch(remaining_samples)
                yield batch

    def prepare_batch(self, samples):
        # Convert lists of samples into tensors
        ae_latent_shape = tuple(samples[0]['ae_latent_shape'])

        siglip_emb = torch.tensor(np.stack([np.frombuffer(s['siglip_emb'], dtype=np.float16).copy() for s in samples]))
        siglip_vec = torch.tensor(np.stack([np.frombuffer(s['siglip_vec'], dtype=np.float16).copy() for s in samples]))
        siglip_max_unpadded = torch.stack([s["siglip_unpadded_len"] for s in samples]).max()
        bert_emb = torch.tensor(np.stack([np.frombuffer(s['bert_emb'], dtype=np.float16).copy() for s in samples]))
        bert_vec = torch.tensor(np.stack([np.frombuffer(s['bert_vec'], dtype=np.float16).copy() for s in samples]))
        bert_max_unpadded = torch.stack([s["bert_unpadded_len"] for s in samples]).max()

        batch = {
            'caption': [s['caption'] for s in samples],
            'ae_latent': torch.tensor(np.stack([np.frombuffer(s['ae_latent'], dtype=np.float16).copy() for s in samples])).reshape(-1, *ae_latent_shape),
            'ae_latent_shape': ae_latent_shape,
            'siglip_emb': siglip_emb[:, :siglip_max_unpadded, :],
            'siglip_vec': siglip_vec,
            'bert_emb': bert_emb[:, :bert_max_unpadded, :],
            'bert_vec': bert_vec,
        }
        return batch
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
