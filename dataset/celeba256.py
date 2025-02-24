class MemmapDataset(Dataset):
    def __init__(self, data_path):
        # Load the entire memmap dataset once
        self.data = torch.load(data_path, map_location='cpu', weights_only=True, mmap=True)

    def __len__(self):
        # Return the total number of samples
        return self.data.shape[0]

    def __getitem__(self, index):
        # Retrieve a single sample by index
        sample = self.data[index]
        
        return sample

def batch_to_tensors(batch):
    latents = batch["latent"]
    latents = torch.stack(
    [torch.stack([torch.stack(inner) for inner in outer]) for outer in latents]
    )
    latents = latents.permute(3, 0, 1, 2) # for some reason batch size is last so we need to permute it
    return latents