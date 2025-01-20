from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Dataset
from transformer.microdit import ReiMei, ReiMeiParameters
from accelerate import Accelerator
from config import BS, MASK_RATIO, VAE_SCALING_FACTOR, VAE_CHANNELS, DS_DIR_BASE
from transformer.utils import random_mask, apply_mask_to_tensor
from datasets import config as hf_config
import numpy as np

# Disable HF online checks if you have the dataset offline
hf_config.HF_HUB_OFFLINE = 1

DTYPE = torch.bfloat16

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
    Returns (lrs, losses) for optional plotting.
    """
    # Copy model so we don't affect main weights
    test_model = deepcopy(model)
    test_model.train()
    # Re-init optimizer to ensure clean state
    optimizer_copy = type(optimizer)(test_model.parameters(), lr=init_lr)

    # Decide how many steps total
    if num_steps is None:
        num_steps = len(train_dataloader)

    # Calculate the multiplier per step
    lr_mult = (final_lr / init_lr) ** (1 / num_steps)
    lr = init_lr

    avg_loss = 0.0
    best_loss = float("inf")
    
    lrs = []
    losses = []

    step_count = 0
    for batch_idx, latents in enumerate(train_dataloader):
        if step_count >= num_steps:
            break

        # Set current LR
        for param_group in optimizer_copy.param_groups:
            param_group['lr'] = lr
        
        latents = latents.to(device, dtype=dtype)
        bs, c, h, w = latents.shape

        # Null text embeddings for your dataset
        cond_embed_dim = 1
        caption_embeddings = torch.zeros((bs, cond_embed_dim), device=device, dtype=dtype)

        # Scale latents
        latents_scaled = latents * VAE_SCALING_FACTOR

        # Random mask
        mask = random_mask(bs, latents_scaled.shape[-2], latents_scaled.shape[-1], mask_ratio=MASK_RATIO).to(device, dtype=dtype)

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
        # Weighted by the unmasked proportion
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

        # Stop if loss diverges too much
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

    # Model params
    base_dim = 1024
    base_heads = 16
    embed_dim = 1024
    num_heads = 16
    num_layers = 4
    mlp_dim = embed_dim
    cond_embed_dim = 1
    num_experts = 1
    active_experts = 1.0
    shared_experts = None
    token_mixer_layers = 2
    dropout = 0.1

    m_d = float(embed_dim) / float(base_dim)
    assert (embed_dim // num_heads) == (base_dim // base_heads)

    # Build ReiMei model
    params = ReiMeiParameters(
        channels=VAE_CHANNELS,
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
        m_d=m_d,
    )

    model = ReiMei(params).to(DTYPE)

    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    # Dataset & DataLoader
    dataset_path = f"{DS_DIR_BASE}/celeb-a-hq-dc-ae-256/latents.pth"
    train_dataset = MemmapDataset(dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=0)

    # Prepare model & data with accelerator
    model, train_dataloader = accelerator.prepare(model, train_dataloader)

    # Create an optimizer (we won't do normal multi-epoch training here)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)

    # Configure LR range test
    init_lr = 1e-7
    final_lr = 1e-2
    num_steps = 500  # e.g., do 500 steps

    print(f"Running LR range test from {init_lr} to {final_lr} for {num_steps} steps.")
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

    # Only save plot on main process
    if accelerator.is_main_process:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Smoothed Loss")
        plt.title(f"LR Range Test (embed_dim={embed_dim}, num_layers={num_layers})")

        plot_filename = f"lr_range_test_embed_{embed_dim}_layers_{num_layers}.png"
        plt.savefig(plot_filename)
        plt.close()

        print("\nLR range test complete. Saved plot as:", plot_filename)
        print("Inspect the curve and pick an LR below the point it starts to diverge.")
