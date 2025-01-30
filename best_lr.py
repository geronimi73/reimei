from copy import deepcopy
import torch
import torch.nn.functional as F
from accelerate import Accelerator
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# From your code/config
from config import (
    BERT_EMBED_DIM,
    SIGLIP_EMBED_DIM,
    BS,
    SEED,
    AE_SCALING_FACTOR,
    AE_CHANNELS,
    DS_DIR_BASE,
    USERNAME,
    DATASET_NAME,
    MASK_RATIO,
)
from datasets import load_dataset
from dataset.shapebatching_dataset import ShapeBatchingDataset
from transformer.reimei import ReiMei, ReiMeiParameters
from transformer.utils import random_mask, apply_mask_to_tensor

DTYPE = torch.bfloat16

def get_dataset(bs, seed, num_workers=16):
    """
    Same dataset-loading logic as in your training script.
    """
    dataset = load_dataset(
        f"{USERNAME}/{DATASET_NAME}",
        cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}",
        split="train",
        num_proc=num_workers
    ).to_iterable_dataset(1000).shuffle(seed, buffer_size=bs * 20)
    dataset = ShapeBatchingDataset(dataset, bs, True, seed)
    return dataset

def lr_range_test(
    accelerator,
    model,
    train_dataloader,
    init_lr=1e-7,
    final_lr=5e-3,
    num_steps=3000,
    beta=0.98,  # smoothing factor for the loss
):
    """
    Exponential LR range test (aka LR finder).
    Returns (lrs, losses).
    Uses Accelerate for backward pass and device management.
    """
    device = accelerator.device

    # Copy model so we don't affect the original weights
    test_model = deepcopy(model)
    test_model.train()

    # Fresh optimizer so it starts from scratch
    optimizer = torch.optim.AdamW(test_model.parameters(), lr=init_lr)

    # Wrap model, optimizer, and dataloader with accelerator
    test_model, optimizer, train_dataloader = accelerator.prepare(
        test_model, optimizer, train_dataloader
    )

    # Exponential LR multiplier
    lr_mult = (final_lr / init_lr) ** (1 / num_steps)
    lr = init_lr

    avg_loss = 0.0
    best_loss = float("inf")

    lrs = []
    losses = []

    step_count = 0

    # Main loop
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="LR Finder")):
        if step_count >= num_steps:
            break

        # Set current LR
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Move data to correct device/dtype
        latents = batch["ae_latent"].to(device, dtype=DTYPE)
        siglip_emb = batch["siglip_emb"].to(device, dtype=DTYPE)
        siglip_vec = batch["siglip_vec"].to(device, dtype=DTYPE)
        bert_emb = batch["bert_emb"].to(device, dtype=DTYPE)
        bert_vec = batch["bert_vec"].to(device, dtype=DTYPE)

        bs, c, h, w = latents.shape

        # Scale latents
        latents_scaled = latents * AE_SCALING_FACTOR

        # Create a random mask
        mask = random_mask(bs, latents_scaled.shape[-2], latents_scaled.shape[-1],
                           mask_ratio=MASK_RATIO).to(device, dtype=DTYPE)

        # t ~ Uniform(0, 1)
        t = torch.rand((bs,), device=device, dtype=DTYPE)
        texp = t.view(bs, 1, 1, 1)

        # Noisy input for training
        z = torch.randn_like(latents_scaled, device=device, dtype=DTYPE)
        x_t = texp * latents_scaled + (1 - texp) * z

        # Forward pass
        vtheta = test_model(
            x_t, t, 
            siglip_emb, siglip_vec, 
            bert_emb, bert_vec, 
            mask
        )

        # Apply the mask to each relevant tensor
        latents_scaled = apply_mask_to_tensor(latents_scaled, mask)
        z = apply_mask_to_tensor(z, mask)
        vtheta = apply_mask_to_tensor(vtheta, mask)

        # v = original - noise
        v = latents_scaled - z

        mse = ((vtheta - v) ** 2).mean()
        loss = mse / (1.0 - MASK_RATIO)  # scale by inverse keep-ratio

        # Backprop
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        # Smooth the loss for stability
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (batch_idx + 1))

        # Track best loss
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        lrs.append(lr)
        losses.append(smoothed_loss)

        # Stop if the loss diverges too much
        if batch_idx > 1 and smoothed_loss > 4 * best_loss:
            print("Loss diverged; stopping early.")
            break

        # Update LR
        lr *= lr_mult
        step_count += 1

    return lrs, losses

if __name__ == "__main__":
    accelerator = Accelerator()
    device = accelerator.device

    # Fixed model hyperparams (same as your training script)
    num_layers = 24
    embed_dim = 1152
    num_heads = embed_dim // 32
    mlp_dim = embed_dim
    dropout = 0.1

    # We only vary num_experts here
    experts_list = [8, 16, 32, 64]

    # Build dataset / dataloader
    dataset = get_dataset(BS, SEED + accelerator.process_index, num_workers=32)
    # Just pass dataset along; we'll wrap it with accelerator in lr_range_test
    train_dataloader = dataset

    # Prepare a dictionary to store results
    curves_dict = {}

    # LR finder range
    init_lr = 1e-7
    final_lr = 5e-3
    num_steps = 3000  # how many steps of the LR test

    for nx in experts_list:
        print(f"\n=== LR Range Test for num_layers={num_layers}, num_experts={nx} ===")

        # Build a fresh ReiMei model
        params = ReiMeiParameters(
            channels=AE_CHANNELS,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            siglip_dim=SIGLIP_EMBED_DIM,
            bert_dim=BERT_EMBED_DIM,
            num_experts=nx,
            active_experts=2.0,  # match your defaults
            shared_experts=1,
            dropout=dropout,
            token_mixer_layers=2,
            image_text_expert_ratio=nx//4,
        )
        model = ReiMei(params).to(DTYPE)

        # Run the LR range test
        lrs, losses = lr_range_test(
            accelerator=accelerator,
            model=model,
            train_dataloader=train_dataloader,
            init_lr=init_lr,
            final_lr=final_lr,
            num_steps=num_steps,
        )
        curves_dict[nx] = (lrs, losses)

    # Plot all combos on one graph (main process only)
    if accelerator.is_main_process:
        os.makedirs("lr_graphs", exist_ok=True)
        plt.figure()

        for nx, (lrs, losses) in curves_dict.items():
            plt.plot(lrs, losses, label=f"Experts={nx}")

        plt.xscale("log")
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Smoothed Loss")

        # Adjust as desired
        plt.ylim(bottom=1.2, top=1.6)
        plt.title(f"LR Range Test (num_layers={num_layers}, varying num_experts)")
        plt.legend()

        plot_filename = "lr_graphs/lr_experts.png"
        plt.savefig(plot_filename)
        plt.close()

        print(f"\nAll LR range tests complete. Combined plot saved to: {plot_filename}")
