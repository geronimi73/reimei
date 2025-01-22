from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Dataset
from transformer.microdit import ReiMei, ReiMeiParameters
from accelerate import Accelerator
from config import BS, EPOCHS, MASK_RATIO, AE_SCALING_FACTOR, AE_CHANNELS, AE_HF_NAME, MODELS_DIR_BASE, DS_DIR_BASE, SEED, USERNAME, DATASET_NAME
from transformer.utils import random_mask, apply_mask_to_tensor
from datasets import config as hf_config
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

hf_config.HF_HUB_OFFLINE = 1  # Disable HF checks if dataset is local

DTYPE = torch.bfloat16

def batch_to_tensors(batch):
    latents = batch["latent"]
    latents = torch.stack(
    [torch.stack([torch.stack(inner) for inner in outer]) for outer in latents]
    )
    latents = latents.permute(3, 0, 1, 2) # for some reason batch size is last so we need to permute it
    return latents

class MemmapDataset(Dataset):
    def __init__(self, data_path):
        # Load the entire memmap dataset once
        self.data = torch.load(data_path, map_location='cpu', weights_only=True, mmap=True)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        return self.data[index]

def lr_range_test(
    model,
    train_dataloader,
    optimizer,
    init_lr=1e-7,
    final_lr=1e-1,
    num_steps=None,
    beta=0.98,  # smoothing factor for the loss
    device="cuda",
    dtype=torch.bfloat16,
):
    """
    Exponential LR range test (aka LR finder).
    Returns (lrs, losses).
    """
    # Copy model so we don't affect the main weights
    test_model = deepcopy(model)
    test_model.train()

    # Re-init the optimizer to ensure a clean state
    optimizer_copy = type(optimizer)(test_model.parameters(), lr=init_lr)

    if num_steps is None:
        num_steps = len(train_dataloader)

    # Exponential LR multiplier
    lr_mult = (final_lr / init_lr) ** (1 / num_steps)
    lr = init_lr

    avg_loss = 0.0
    best_loss = float("inf")
    
    lrs = []
    losses = []

    step_count = 0
    for batch_idx, latents in tqdm(enumerate(train_dataloader)):
        if step_count >= num_steps:
            break

        # Set current LR
        for param_group in optimizer_copy.param_groups:
            param_group['lr'] = lr
        
        latents = batch_to_tensors(latents).to(device, DTYPE)
        # latents = latents.to(device, dtype=dtype)
        bs, c, h, w = latents.shape

        # Null text embeddings
        cond_embed_dim = 1
        caption_embeddings = torch.zeros((bs, cond_embed_dim), device=device, dtype=dtype)

        # Scale latents
        latents_scaled = latents * AE_SCALING_FACTOR

        # Random mask
        mask = random_mask(bs, latents_scaled.shape[-2], latents_scaled.shape[-1],
                           mask_ratio=MASK_RATIO).to(device, dtype=dtype)

        # Time conditioning
        nt = torch.randn((bs,), device=device, dtype=dtype)
        t = torch.sigmoid(nt)
        texp = t.view([bs, 1, 1, 1])
        z1 = torch.randn_like(latents_scaled, device=device, dtype=dtype)
        zt = (1 - texp) * latents_scaled + texp * z1

        # Forward
        vtheta = test_model(zt, t, caption_embeddings.unsqueeze(1), caption_embeddings, mask)
        sample_theta = z1 - vtheta

        # Apply mask
        latents_masked = apply_mask_to_tensor(latents_scaled, mask)
        sample_theta_masked = apply_mask_to_tensor(sample_theta, mask)

        batchwise_mse = ((sample_theta_masked - latents_masked) ** 2).mean()
        loss = batchwise_mse * (1 / (1 - MASK_RATIO))

        # Backprop
        optimizer_copy.zero_grad()
        loss.backward()
        optimizer_copy.step()

        # Smooth the loss for stability
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**(batch_idx + 1))

        # Track best loss
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        lrs.append(lr)
        losses.append(smoothed_loss)

        # Stop if the loss diverges too much
        if (batch_idx > 1 and smoothed_loss > 4 * best_loss):
            print("Loss diverged; stopping early.")
            break

        # Update LR for next step
        lr *= lr_mult
        step_count += 1

    return lrs, losses

if __name__ == "__main__":
    accelerator = Accelerator()
    device = accelerator.device

    # ----------------- Common Hyperparams ------------------
    base_dim = 1024
    base_heads = 16
    embed_dim = 1024       # Keep embed_dim fixed here
    num_heads = 16         # Since embed_dim=1024 and base_dim=1024, head_dim = 64
    mlp_dim = embed_dim
    cond_embed_dim = 1
    dropout = 0.1
    token_mixer_layers = 2

    # The sets of hyperparameters we want to compare
    # layers_list = [2, 4]
    # experts_list = [4, 8, 16]
    layers_list = [2]
    experts_list = [2]

    # Dataset & DataLoader
    # dataset_path = f"{DS_DIR_BASE}/celeb-a-hq-dc-ae-256/latents.pth"
    # train_dataset = MemmapDataset(dataset_path)
    train_dataset = load_dataset(f"{USERNAME}/{DATASET_NAME}", split="train", cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}")
    train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=0)
    train_dataloader = accelerator.prepare(train_dataloader)

    # We'll store LR/loss curves for each (num_layers, num_experts) combo
    curves_dict = {}  # { (layers, experts): (lrs, losses) }

    # LR range test parameters
    init_lr = 1e-7
    final_lr = 5e-3
    num_steps = 300

    m_d = float(embed_dim) / float(base_dim)
    # The assertion must still hold: (embed_dim // num_heads) == (base_dim // base_heads)

    for nl in layers_list:
        for nx in experts_list:
            print(f"\n=== LR Range Test for num_layers={nl}, num_experts={nx} ===")

            # Build a fresh model
            from transformer.microdit import ReiMeiParameters
            params = ReiMeiParameters(
                channels=AE_CHANNELS,
                embed_dim=embed_dim,
                num_layers=nl,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                text_embed_dim=cond_embed_dim,
                vector_embed_dim=cond_embed_dim,
                num_experts=nx,
                active_experts=1.0,
                shared_experts=None,
                dropout=dropout,
                token_mixer_layers=token_mixer_layers,
                m_d=m_d,
            )
            model = ReiMei(params).to(DTYPE)
            model = accelerator.prepare(model)

            # New optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)

            lrs, losses = lr_range_test(
                model,
                train_dataloader,
                optimizer,
                init_lr=init_lr,
                final_lr=final_lr,
                num_steps=num_steps,
                device=device,
                dtype=DTYPE,
            )
            curves_dict[(nl, nx)] = (lrs, losses)

    # ----------------- Plot all combos on one graph ------------------
    if accelerator.is_main_process:
        import matplotlib.pyplot as plt
        plt.figure()

        for (nl, nx), (lrs, losses) in curves_dict.items():
            label_str = f"L={nl}, Experts={nx}"
            plt.plot(lrs, losses, label=label_str)
        
        plt.xscale("log")
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Smoothed Loss")
        plt.ylim(bottom=1.2, top=1.6)
        plt.title("LR Range Test: layers vs. num_experts (embed_dim=1024)")
        plt.legend()

        import os
        os.makedirs("lr_graphs", exist_ok=True)
        plot_filename = "lr_graphs/lr_layers_vs_experts_1024.png"
        plt.savefig(plot_filename)
        plt.close()

        print(f"\nAll LR range tests complete. Combined plot saved to: {plot_filename}")
        print("All lines are in one graph with y-limits from 1.65 to 2.2.")
