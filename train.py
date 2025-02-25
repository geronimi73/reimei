from copy import deepcopy
import json
import random
import torch
from dataset.shapebatching_dataset import ShapeBatchingDataset, get_dataset
from dataset.inet96 import ImageNet96Dataset
from transformer.reimei import ReiMei, ReiMeiParameters
from accelerate import Accelerator
from config import BERT_EMBED_DIM, BERT_HF_NAME, BS, CFG_RATIO, MAX_CAPTION_LEN, TRAIN_STEPS, MASK_RATIO, AE_SCALING_FACTOR, AE_CHANNELS, AE_HF_NAME, MODELS_DIR_BASE, DS_DIR_BASE, SEED, SIGLIP_EMBED_DIM, SIGLIP_HF_NAME, USERNAME, DATASET_NAME, LR
from config import DIT_S as DIT
from datasets import load_dataset
from transformer.utils import random_cfg_mask, random_mask, apply_mask_to_tensor
from tqdm import tqdm
import datasets
import torchvision
import os
import pickle
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from diffusers import AutoencoderDC
from diffusers import AutoencoderKL
from torch.optim.lr_scheduler import OneCycleLR
import wandb
from torchmetrics.functional.multimodal import clip_score
from transformers import SiglipModel, SiglipProcessor

DTYPE = torch.bfloat16

@torch.no_grad
def sample_images(model, vae, ds, noise, prompts, sig_emb, sig_vec, bert_emb, bert_vec):
    def normalize_batch(images):
        min_vals = images.amin(dim=(1, 2, 3), keepdim=True)  # Min per image
        max_vals = images.amax(dim=(1, 2, 3), keepdim=True)  # Max per image
        
        # Ensure no division by zero
        scale = (max_vals - min_vals).clamp(min=1e-8)
        
        return (images - min_vals) / scale

    # Use the stored embeddings
    sampled_latents = model.sample(noise, sig_emb, sig_vec, bert_emb, bert_vec, sample_steps=50, cfg=1.0).to(device, dtype=DTYPE)
    cfg_sampled_latents = model.sample(noise, sig_emb, sig_vec, bert_emb, bert_vec, sample_steps=50, cfg=3.0).to(device, dtype=DTYPE)
    
    # Decode latents to images
    sampled_images = normalize_batch(vae.decode(sampled_latents).sample)
    cfg_sampled_images = normalize_batch(vae.decode(cfg_sampled_latents).sample)

    # Compute SigLIP scores
    scores = calculate_score(sampled_images, prompts)
    cfg_scores = calculate_score(cfg_sampled_images, prompts)

    # Log the sampled images
    interleaved = torch.stack([sampled_images, cfg_sampled_images], dim=1).reshape(-1, *sampled_images.shape[1:])

    grid = torchvision.utils.make_grid(interleaved, nrow=2)

    return grid, scores, cfg_scores

if __name__ == "__main__":
    # Comment this out if you havent downloaded dataset and models yet
    datasets.config.HF_HUB_OFFLINE = 1
    # torch.set_float32_matmul_precision('high')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    embed_dim = 768
    patch_size = (1,1)

    params = ReiMeiParameters(
        use_mmdit=True,
        use_ec=False,
        channels=AE_CHANNELS,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_layers=4,
        num_heads=(embed_dim // 128),
        siglip_dim=SIGLIP_EMBED_DIM,
        bert_dim=BERT_EMBED_DIM,
        num_experts=32,
        capacity_factor=4.0,
        shared_experts=1,
        dropout=0.1,
        token_mixer_layers=1,
        image_text_expert_ratio=2,
    )

    accelerator = Accelerator()
    device = accelerator.device

    model = ReiMei(params)
    # model = torch.compile(ReiMei(params))

    params_count = sum(p.numel() for p in model.parameters())
    print("Number of parameters: ", params_count)

    wandb.init(project="ReiMei", config={
        "params_count": params_count,
        "dataset_name": DATASET_NAME,
        "ae_hf_name": AE_HF_NAME,
        "lr": LR,
        "bs": BS,
        "CFG_RATIO": CFG_RATIO,
        "MASK_RATIO": MASK_RATIO,
        "MAX_CAPTION_LEN": MAX_CAPTION_LEN,
        "params": params,
    })
    
    # dataset = get_dataset(BS, SEED + accelerator.process_index, device=device, dtype=DTYPE, num_workers=1)
    # ds = MemmapDataset(f"{DS_DIR_BASE}/celeb-a-hq-dc-ae-256/latents.pth")
    # ds = load_dataset(f"{USERNAME}/{DATASET_NAME}", cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}", num_proc=16, split="train").to_iterable_dataset(1000)
    # ds = ImageNetDataset(f"{DS_DIR_BASE}/imagenet.int8/inet.npy", f"{DS_DIR_BASE}/imagenet.int8/inet.json")
    # dataset = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=1)
    # dataset = InfiniteDataLoader(dataset)

    ds = load_dataset(f"{DATASET_NAME}", cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}", num_proc=1, split="train").to_iterable_dataset(1000)

    siglip_model = SiglipModel.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip").to(device, DTYPE)
    siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip")

    @torch.no_grad
    def encode_siglip(captions):
        # Encode the captions
        s_tokens = siglip_processor.tokenizer(captions, padding='longest', truncation=True, return_tensors="pt", max_length=MAX_CAPTION_LEN).to(device)
        siglip_outputs = siglip_model.text_model(**s_tokens, output_hidden_states=True)
        siglip_embedding = siglip_outputs.hidden_states[-1]
        siglip_vec = siglip_outputs.pooler_output

        return siglip_embedding, siglip_vec
    
    @torch.no_grad
    def calculate_score(images, captions):
        # Tokenize captions
        text_inputs = siglip_processor.tokenizer(
            captions, padding='longest', truncation=True, return_tensors="pt", max_length=MAX_CAPTION_LEN
        ).to(device)
        
        # Encode text
        text_embeddings = siglip_model.get_text_features(**text_inputs, output_hidden_states=True)
        
        # Encode images
        image_inputs = siglip_processor(images=images.to(torch.float32), return_tensors="pt").to(device)
        image_outputs = siglip_model.get_image_features(**image_inputs)  # [B, D]
        
        # Compute cosine similarity
        scores = torch.nn.functional.cosine_similarity(image_outputs, text_embeddings, dim=-1)
        return scores

    ds = ImageNet96Dataset(ds, BS, device, DTYPE)


    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=LR, weight_decay=0.01)

    # scheduler = OneCycleLR(optimizer, max_lr=LR, total_steps=TRAIN_STEPS)

    # model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, dataset)
    model, optimizer, ds = accelerator.prepare(model, optimizer, ds)

    # checkpoint = torch.load(f"models/pretrained_reimei_model_and_optimizer.pt")
    # checkpoint = torch.load(f"models/reimei_model_and_optimizer_3_f32.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # del checkpoint
    
    if accelerator.is_main_process:
        ae = AutoencoderDC.from_pretrained(f"mit-han-lab/{AE_HF_NAME}", torch_dtype=DTYPE, cache_dir=f"{MODELS_DIR_BASE}/dc_ae", revision="main").to(device).eval()
        # ae =  AutoencoderKL.from_pretrained(f"{AE_HF_NAME}", cache_dir=f"{MODELS_DIR_BASE}/vae").to(device=device, dtype=DTYPE).eval()
        assert ae.config.scaling_factor == AE_SCALING_FACTOR, f"Scaling factor mismatch: {ae.config.scaling_factor} != {AE_SCALING_FACTOR}"
        
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        noise = torch.randn(4, AE_CHANNELS, 3, 3).to(device, dtype=DTYPE)
        example_batch = next(iter(ds))

        # example_latents = example_batch.to(device, dtype=DTYPE)[:4]

        example_latents = example_batch["ae_latent"][:4].to(device, dtype=DTYPE)
        # ex_sig_emb = example_batch["siglip_emb"][:4].to(device, dtype=DTYPE)
        # ex_sig_vec = example_batch["siglip_vec"][:4].to(device, dtype=DTYPE)

        example_captions = example_batch["caption"][:4]
        
        with torch.no_grad():
            example_ground_truth = ae.decode(example_latents).sample
        grid = torchvision.utils.make_grid(example_ground_truth, nrow=2, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid, f"logs/example_images.png")

        # Save captions
        with open("logs/example_captions.txt", "w") as f:
            for index, caption in enumerate(example_captions):
                f.write(f"{index}: {caption}\n")

        del grid, example_ground_truth, example_latents

        # example_captions = ["a green field with green bushes", "bright blue sky with clouds", "a red apple on a wooden table", "a field of green grass with a snowcapped mountain in the background"]
        ex_captions = ["a cheeseburger on a white plate", "a bunch of bananas on a wooden table", "a white tea pot on a wooden table", "an erupting volcano with lava pouring out"]
        ex_sig_emb, ex_sig_vec= encode_siglip(ex_captions)

        # ex_sig_emb = torch.zeros(4, 1, 1152).to(device, dtype=DTYPE)
        # ex_sig_vec = torch.zeros(4, 1152).to(device, dtype=DTYPE)
        ex_bert_emb = torch.zeros(4, 1, 1024).to(device, dtype=DTYPE)
        ex_bert_vec = torch.zeros(4, 1024).to(device, dtype=DTYPE)

        ae = ae.to("cpu")

    print("Starting training...")

    progress_bar = tqdm(ds, leave=False, total=TRAIN_STEPS)
    for batch_idx, batch in enumerate(progress_bar):
        # latents = batch_to_tensors(batch).to(device, DTYPE)
        # latents = batch.to(device, dtype=DTYPE)

        latents = batch["ae_latent"].to(device, dtype=DTYPE)

        bs, c, h, w = latents.shape
        latents = latents * AE_SCALING_FACTOR
        # siglip_emb = torch.zeros(bs, 1, 1152).to(device, dtype=DTYPE)
        # siglip_vec = torch.zeros(bs, 1152).to(device, dtype=DTYPE)
        bert_emb = torch.zeros(bs, 1, 1024).to(device, dtype=DTYPE)
        bert_vec = torch.zeros(bs, 1024).to(device, dtype=DTYPE)

        # img_mask = random_mask(bs, latents.shape[-2], latents.shape[-1], patch_size, mask_ratio=MASK_RATIO).to(device, dtype=DTYPE)
        cfg_mask = random_cfg_mask(bs, 0.1).to(device, dtype=DTYPE)

        siglip_emb, siglip_vec = encode_siglip(batch["caption"])

        siglip_emb = siglip_emb.to(device, dtype=DTYPE) * cfg_mask.view(bs, 1, 1)
        siglip_vec = siglip_vec.to(device, dtype=DTYPE) * cfg_mask.view(bs, 1)
        # bert_emb = batch["bert_emb"].to(device, dtype=DTYPE) * cfg_mask.view(bs, 1, 1)
        # bert_vec = batch["bert_vec"].to(device, dtype=DTYPE) * cfg_mask.view(bs, 1)

        # txt_mask = random_mask(bs, siglip_emb.size(1), 1, (1, 1), mask_ratio=0).to(device=device, dtype=DTYPE)
        txt_mask = random_mask(bs, siglip_emb.size(1), 1, (1, 1), mask_ratio=MASK_RATIO).to(device=device, dtype=DTYPE)
        # txt_mask = random_mask(bs, siglip_emb.size(1)+bert_emb.size(1), 1, (1, 1), mask_ratio=MASK_RATIO).to(device=device, dtype=DTYPE)

        nt = torch.randn((bs,), device=device, dtype=DTYPE)
        t = torch.sigmoid(nt)
        texp = t.view([bs, 1, 1, 1]).to(device, dtype=DTYPE)

        z = torch.randn_like(latents, device=device, dtype=DTYPE)
        x_t = (1 - texp) * latents + texp * z

        vtheta = model(x_t, t, siglip_emb, siglip_vec, bert_emb, bert_vec, None, txt_mask)

        # vtheta = apply_mask_to_tensor(vtheta, img_mask, patch_size)
        # latents = apply_mask_to_tensor(latents, img_mask, patch_size)
        # z = apply_mask_to_tensor(z, img_mask, patch_size)

        v = z - latents

        mse = (((v - vtheta) ** 2)).mean()
        # loss = mse.mean() * (1 / (1 - MASK_RATIO))
        loss = mse

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        # scheduler.step()

        progress_bar.set_postfix(loss=loss.item())
        
        if accelerator.is_main_process:
            wandb.log({"loss": loss.item()}, step=batch_idx)

        del mse, loss, vtheta

        if batch_idx % 1000 == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                model.eval()
                ae = ae.to(device)

                grid, scores, cfg_scores = sample_images(model, ae, ds, noise, ex_captions, ex_sig_emb, ex_sig_vec, ex_bert_emb, ex_bert_vec)
                torchvision.utils.save_image(grid, f"logs/sampled_images_step_{batch_idx}.png")

                wandb.log({"Siglip scores": scores.mean().item(), "Siglip scores with CFG": cfg_scores.mean().item()}, step=batch_idx)

                del grid, scores, cfg_scores

                ae = ae.to("cpu")

                model.train()

        if ((batch_idx % (TRAIN_STEPS//10)) == 0) and batch_idx != 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_optimizer = accelerator.unwrap_model(optimizer)
                model_save_path = f"models/reimei_model_and_optimizer_{batch_idx//(TRAIN_STEPS//10)}_f32.pt"
                torch.save({
                    'global_step': batch_idx,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                }, model_save_path)
                print(f"Model saved to {model_save_path}.")
        
        if batch_idx == TRAIN_STEPS - 1:
            print("Training complete.")

            # Save model in /models
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_optimizer = accelerator.unwrap_model(optimizer)
                model_save_path = "models/pretrained_reimei_model_and_optimizer.pt"
                torch.save(
                    {
                        'global_step': batch_idx,
                        'model_state_dict': unwrapped_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict(),
                    },
                    model_save_path,
                )
                print(f"Model saved to {model_save_path}.")

            break