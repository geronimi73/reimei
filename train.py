from copy import deepcopy
import torch
from dataset.shapebatching_dataset import ShapeBatchingDataset
from transformer.reimei import ReiMei, ReiMeiParameters
from accelerate import Accelerator
from config import BS, EPOCHS, MASK_RATIO, AE_SCALING_FACTOR, AE_CHANNELS, AE_HF_NAME, MODELS_DIR_BASE, DS_DIR_BASE, SEED, USERNAME, DATASET_NAME, LR
from config import DIT_S as DIT
from datasets import load_dataset
from transformer.utils import random_mask, apply_mask_to_tensor
from tqdm import tqdm
import datasets
import torchvision
import os
import pickle
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from diffusers import AutoencoderDC
from torch.optim.lr_scheduler import OneCycleLR

DTYPE = torch.bfloat16

def sample_images(model, vae, noise, sig_emb, sig_vec, bert_emb, bert_vec):
    with torch.no_grad():
        # Use the stored embeddings
        sampled_latents = model.sample(noise, sig_emb, sig_vec, bert_emb, bert_vec, sample_steps=50)
        
        # Decode latents to images
        sampled_images = vae.decode(sampled_latents).sample

    # Log the sampled images
    grid = torchvision.utils.make_grid(sampled_images, nrow=3, normalize=True, scale_each=True)
    return grid

def get_dataset(bs, seed, num_workers=16):
    dataset = load_dataset(f"{USERNAME}/{DATASET_NAME}", cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}", split="train").to_iterable_dataset(1000).shuffle(seed, buffer_size = bs * 20)
    dataset = ShapeBatchingDataset(dataset, bs, True, seed)
    return dataset

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

def update_ema(ema_model, model, decay):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())

    for name in ema_params.keys():
        ema_params[name].data.mul_(decay).add_(model_params[name].data, alpha=1 - decay)

if __name__ == "__main__":
    # Comment this out if you havent downloaded dataset and models yet
    datasets.config.HF_HUB_OFFLINE = 1

    base_dim = 1024
    base_heads = 16

    input_dim = AE_CHANNELS
    embed_dim = 1024
    num_layers = 28
    num_heads = 16
    mlp_dim = embed_dim
    cond_embed_dim = 1 # Null for this dataset
    # pos_embed_dim = 60
    pos_embed_dim = None
    num_experts = 32
    image_text_expert_ratio = 8
    active_experts = 2.0
    shared_experts = None
    token_mixer_layers = 1
    dropout = 0.1

    m_d = float(embed_dim) / float(base_dim)

    assert (embed_dim // num_heads) == (base_dim // base_heads)

    params = ReiMeiParameters(
        channels=input_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        text_embed_dim=cond_embed_dim,
        vector_embed_dim=cond_embed_dim,
        num_experts=num_experts,
        active_experts=active_experts,
        shared_experts=shared_experts,
        dropout=dropout,
        token_mixer_layers=token_mixer_layers,
        image_text_expert_ratio=image_text_expert_ratio,
        m_d=m_d,
    )

    accelerator = Accelerator()
    device = accelerator.device

    model = ReiMei(params).to(DTYPE)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    print("Starting training...")
    
    dataset = get_dataset(BS, SEED + accelerator.process_index, num_workers=64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(dataset), epochs=EPOCHS)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, dataset)

    # checkpoint = torch.load(f"models/reimei_model_and_optimizer_epoch_0_f32.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # del checkpoint
    
    if accelerator.is_main_process:
        dc_ae = AutoencoderDC.from_pretrained(f"mit-han-lab/{AE_HF_NAME}", torch_dtype=DTYPE, cache_dir=f"{MODELS_DIR_BASE}/dc_ae", revision="main").to(device).eval()
        assert dc_ae.config.scaling_factor == AE_SCALING_FACTOR, f"Scaling factor mismatch: {dc_ae.config.scaling_factor} != {AE_SCALING_FACTOR}"
        
        os.makedirs("logs", exist_ok=True)

        noise = torch.randn(9, AE_CHANNELS, 16, 16).to(device, dtype=DTYPE)
        example_batch = next(iter(dataset))
        ex_sig_emb = example_batch["siglip_emb"][:9].to(device, dtype=DTYPE)
        ex_sig_vec = example_batch["siglip_vec"][:9].to(device, dtype=DTYPE)
        ex_bert_emb = example_batch["bert_emb"][:9].to(device, dtype=DTYPE)
        ex_bert_vec = example_batch["bert_vec"][:9].to(device, dtype=DTYPE)
        example_latents = example_batch["ae_latent"][:9].to(device, dtype=DTYPE)
        example_captions = example_batch["caption"][:9]

        with torch.no_grad():
            example_ground_truth = dc_ae.decode(example_latents).sample
        grid = torchvision.utils.make_grid(example_ground_truth, nrow=3, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid, f"logs/example_images.png")

        # Save captions
        with open("logs/example_captions.txt", "w") as f:
            for index, caption in enumerate(example_captions):
                f.write(f"{index}: {caption}\n")
        dc_ae = dc_ae.to("cpu")
        losses = []

        del example_batch, ex_sig_emb, ex_sig_vec, ex_bert_emb, ex_bert_vec, example_captions, example_latents, example_ground_truth, grid

    ema_model = deepcopy(model)
    ema_decay = 0.999

    for epoch in range(EPOCHS):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            # latents = batch_to_tensors(batch).to(device, DTYPE)
            latents = batch["ae_latent"].to(device, dtype=DTYPE)
            siglip_emb = batch["siglip_emb"].to(device, dtype=DTYPE)
            siglip_vec = batch["siglip_vec"].to(device, dtype=DTYPE)
            bert_emb = batch["bert_emb"].to(device, dtype=DTYPE)
            bert_vec = batch["bert_vec"].to(device, dtype=DTYPE)

            bs, c, h, w = latents.shape
            latents = latents * AE_SCALING_FACTOR

            mask = random_mask(bs, latents.shape[-2], latents.shape[-1], mask_ratio=MASK_RATIO).to(device, dtype=DTYPE)

            nt = torch.randn((bs,)).to(device, dtype=DTYPE)
            t = torch.sigmoid(nt)
            
            texp = t.view([bs, *([1] * len(latents.shape[1:]))]).to(device, dtype=DTYPE)
            z1 = torch.randn_like(latents, device=device, dtype=DTYPE)
            zt = (1 - texp) * latents + texp * z1

            vtheta = model(zt, t, siglip_emb, siglip_vec, bert_emb, bert_vec, mask)
            sample_theta = z1 - vtheta

            latents = apply_mask_to_tensor(latents, mask)
            sample_theta = apply_mask_to_tensor(sample_theta, mask)

            batchwise_mse = ((sample_theta - latents) ** 2).mean()
            loss = batchwise_mse * 1 / (1 - MASK_RATIO)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            update_ema(ema_model, model, ema_decay)

            progress_bar.set_postfix(loss=loss.item())
            if accelerator.is_local_main_process:
                losses.append(loss.item())

                if batch_idx % 200 == 0:
                    model.eval()
                    dc_ae = dc_ae.to(device)

                    grid = sample_images(model, dc_ae, noise, ex_sig_emb, ex_sig_vec, ex_bert_emb, ex_bert_vec)
                    torchvision.utils.save_image(grid, f"logs/sampled_images_epoch_{epoch}_batch_{batch_idx}.png")

                    del grid

                    dc_ae = dc_ae.to("cpu")

                    model.train()

        print(f"Epoch {epoch} complete.")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and epoch % 10 == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_optimizer = accelerator.unwrap_model(optimizer)
            model_save_path = f"models/reimei_model_and_optimizer_epoch_{epoch}_f32.pt"
            torch.save({
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
            }, model_save_path)
            print(f"Model saved to {model_save_path}.")

    print("Training complete.")

    # Save model in /models
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save losses as a pickle
        with open("logs/losses.pkl", "wb") as f:
            pickle.dump(losses, f)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_optimizer = accelerator.unwrap_model(optimizer)
        model_save_path = "models/pretrained_reimei_model_and_optimizer.pt"
        torch.save(
            {
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
            },
            model_save_path,
        )
        print(f"Model saved to {model_save_path}.")