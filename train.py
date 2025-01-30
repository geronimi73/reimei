from copy import deepcopy
import torch
from dataset.shapebatching_dataset import ShapeBatchingDataset
from transformer.reimei import ReiMei, ReiMeiParameters
from accelerate import Accelerator
from config import BERT_EMBED_DIM, BERT_HF_NAME, BS, CFG_RATIO, TRAIN_STEPS, MASK_RATIO, AE_SCALING_FACTOR, AE_CHANNELS, AE_HF_NAME, MODELS_DIR_BASE, DS_DIR_BASE, SEED, SIGLIP_EMBED_DIM, SIGLIP_HF_NAME, USERNAME, DATASET_NAME, LR
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
from transformers import SiglipTokenizer, SiglipTextModel, AutoTokenizer, ModernBertModel

DTYPE = torch.bfloat16

def sample_images(model, vae, noise, sig_emb, sig_vec, bert_emb, bert_vec):
    with torch.no_grad():
        # Use the stored embeddings
        # two_step_latents = model.sample(noise, sig_emb, sig_vec, bert_emb, bert_vec, sample_steps=1).to(device, dtype=DTYPE)
        sampled_latents = model.sample(noise, sig_emb, sig_vec, bert_emb, bert_vec, sample_steps=50).to(device, dtype=DTYPE)
        
        # Decode latents to images
        # two_step_images = vae.decode(two_step_latents).sample
        sampled_images = vae.decode(sampled_latents).sample

    # Log the sampled images
    # interleaved = torch.stack([two_step_images, sampled_images], dim=1).view(-1, *two_step_images.shape[1:])
    grid = torchvision.utils.make_grid(sampled_images, nrow=3, normalize=True, scale_each=True)
    return grid

def get_dataset(bs, seed, device, num_workers=16):
    ds = load_dataset(f"{USERNAME}/{DATASET_NAME}", cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}",num_proc=num_workers, split="train")
    ds = ds.to_iterable_dataset(1000).batch(bs*2)
    siglip_model = SiglipTextModel.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip").to(device)
    siglip_tokenizer = SiglipTokenizer.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip")
    bert_model = ModernBertModel.from_pretrained(BERT_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/modernbert").to(device)
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/modernbert")
    
    ds = ShapeBatchingDataset(ds, bs, siglip_tokenizer, siglip_model, bert_tokenizer, bert_model, device, shuffle=True, seed=seed)
    return ds

# class MemmapDataset(Dataset):
#     def __init__(self, data_path):
#         # Load the entire memmap dataset once
#         self.data = torch.load(data_path, map_location='cpu', weights_only=True, mmap=True)

#     def __len__(self):
#         # Return the total number of samples
#         return self.data.shape[0]

#     def __getitem__(self, index):
#         # Retrieve a single sample by index
#         sample = self.data[index]
        
#         return sample

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

    input_dim = AE_CHANNELS
    num_layers = 4
    embed_dim = 1152
    num_heads = embed_dim // 32
    mlp_dim = embed_dim
    num_experts = 2
    active_experts = 1.0
    shared_experts = None
    token_mixer_layers = 2
    image_text_expert_ratio = 1
    dropout = 0.1

    params = ReiMeiParameters(
        channels=input_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        siglip_dim=SIGLIP_EMBED_DIM,
        bert_dim=BERT_EMBED_DIM,
        num_experts=num_experts,
        active_experts=active_experts,
        shared_experts=shared_experts,
        dropout=dropout,
        token_mixer_layers=token_mixer_layers,
        image_text_expert_ratio=image_text_expert_ratio,
    )

    accelerator = Accelerator()
    device = accelerator.device

    model = ReiMei(params).to(DTYPE)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    print("Starting training...")
    
    dataset = get_dataset(BS, SEED + accelerator.process_index, device=device, num_workers=64)
    # dataset = MemmapDataset(f"{DS_DIR_BASE}/celeb-a-hq-dc-ae-256/latents.pth")
    # dataset = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    scheduler = OneCycleLR(optimizer, max_lr=LR, total_steps=TRAIN_STEPS)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, dataset)

    # checkpoint = torch.load(f"models/reimei_model_and_optimizer_0_f32.pt")
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
        # ex_sig_emb = torch.zeros(9, 64, 1152).to(device, dtype=DTYPE)
        # ex_sig_vec = torch.zeros(9, 1152).to(device, dtype=DTYPE)
        # ex_bert_emb = torch.zeros(9, 64, 1024).to(device, dtype=DTYPE)
        # ex_bert_vec = torch.zeros(9, 1024).to(device, dtype=DTYPE)
        # example_latents = example_batch.to(device, dtype=DTYPE)[:9]
        
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

        del example_batch, example_captions, example_latents, example_ground_truth, grid
        # del example_batch, example_latents, example_ground_truth, grid


    ema_decay = 0.999
    progress_bar = tqdm(train_dataloader, leave=False, total=TRAIN_STEPS)
    for batch_idx, batch in enumerate(progress_bar):
        # latents = batch_to_tensors(batch).to(device, DTYPE)
        # latents = batch.to(device, dtype=DTYPE)

        latents = batch["ae_latent"].to(device, dtype=DTYPE)
        siglip_emb = batch["siglip_emb"].to(device, dtype=DTYPE)
        siglip_vec = batch["siglip_vec"].to(device, dtype=DTYPE)
        bert_emb = batch["bert_emb"].to(device, dtype=DTYPE)
        bert_vec = batch["bert_vec"].to(device, dtype=DTYPE)

        bs, c, h, w = latents.shape
        latents = latents * AE_SCALING_FACTOR

        # siglip_emb = torch.zeros(bs, 64, 1152).to(device, dtype=DTYPE)
        # siglip_vec = torch.zeros(bs, 1152).to(device, dtype=DTYPE)
        # bert_emb = torch.zeros(bs, 64, 1024).to(device, dtype=DTYPE)
        # bert_vec = torch.zeros(bs, 1024).to(device, dtype=DTYPE)

        mask = random_mask(bs, latents.shape[-2], latents.shape[-1], mask_ratio=MASK_RATIO).to(device, dtype=DTYPE)

        cfg_mask = random_mask(bs, 1, 1, CFG_RATIO).to(device, dtype=DTYPE).view(bs)
        siglip_emb = siglip_emb * cfg_mask.view(bs, 1, 1)
        siglip_vec = siglip_vec * cfg_mask.view(bs, 1)
        bert_emb = bert_emb * cfg_mask.view(bs, 1, 1)
        bert_vec = bert_vec * cfg_mask.view(bs, 1)

        t = torch.rand((bs,), device=device, dtype=DTYPE)
        texp = t.view([bs, 1, 1, 1]).to(device, dtype=DTYPE)

        z = torch.randn_like(latents, device=device, dtype=DTYPE)
        x_t = (texp * latents) + ((1 - texp) * z)

        vtheta = model(x_t, t, siglip_emb, siglip_vec, bert_emb, bert_vec, mask)

        latents = apply_mask_to_tensor(latents, mask)
        z = apply_mask_to_tensor(z, mask)
        vtheta = apply_mask_to_tensor(vtheta, mask)

        v = latents - z

        mse = ((vtheta - v) ** 2).mean()
        loss = mse * 1 / (1 - MASK_RATIO)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        # Ema only for last 5% of steps
        if batch_idx >= TRAIN_STEPS * 0.95:
            if batch_idx == TRAIN_STEPS * 0.95:
                ema_model = deepcopy(model)
            else:
                update_ema(ema_model, model, ema_decay)

        progress_bar.set_postfix(loss=loss.item())
        if accelerator.is_main_process:
            losses.append(loss.item())

            if batch_idx % 1000 == 0:
                model.eval()
                dc_ae = dc_ae.to(device)

                grid = sample_images(model, dc_ae, noise, ex_sig_emb, ex_sig_vec, ex_bert_emb, ex_bert_vec)
                torchvision.utils.save_image(grid, f"logs/sampled_images_step_{batch_idx}.png")

                del grid

                dc_ae = dc_ae.to("cpu")

                model.train()

            if ((batch_idx % (TRAIN_STEPS//10)) == 0) and batch_idx != 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_optimizer = accelerator.unwrap_model(optimizer)
                model_save_path = f"models/reimei_model_and_optimizer_{batch_idx//(TRAIN_STEPS//10)}_f32.pt"
                torch.save({
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'ema_model_state_dict': ema_model.state_dict(),
                }, model_save_path)
                print(f"Model saved to {model_save_path}.")
        
        if batch_idx == TRAIN_STEPS - 1:
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

            break