import json
import os
import time
import io
import threading
import concurrent.futures
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torchvision import transforms

# --- diffusers, transformers, huggingface_hub imports ---
from diffusers import AutoencoderDC
from transformers import (
    SiglipTokenizer, 
    SiglipTextModel, 
    AutoTokenizer
)
from huggingface_hub import HfFileSystem, hf_hub_download, HfApi

# Your custom or local model classes
from transformers import ModernBertModel

# --------------------------------------------------------------------------------
# --------------------------- Configuration ---------------------------------------
# --------------------------------------------------------------------------------

DATASET = "commoncatalog-cc-by"
DATASET_DIR_BASE = "./datasets"
MODELS_DIR_BASE = "./models"

AE_HF_NAME = "dc-ae-f32c32-mix-1.0-diffusers"
SIGLIP_HF_NAME = "google/siglip-so400m-patch14-384"
BERT_HF_NAME = "answerdotai/ModernBERT-large"

IMAGE_COLUMN_NAME = "jpg"
IMAGE_ID_COLUMN_NAME = "key"
BS = 64

DELETE_AFTER_PROCESSING = True
UPLOAD_TO_HUGGINGFACE = True
USERNAME = "SwayStar123"
PARTITIONS = [1]

# --------------------------------------------------------------------------------
# ---------------------- Utility Classes/Functions --------------------------------
# --------------------------------------------------------------------------------

def get_prng(seed):
    return np.random.RandomState(seed)

class BucketManager:
    """
    Manages 'ideal' bucket resolutions. 
    In the original code, it tries to pick the best (width, height) 
    for an image of a certain aspect ratio, subject to constraints.
    """
    def __init__(
        self,
        max_size=(512,512), 
        divisible=32, 
        min_dim=256, 
        base_res=(512,512),
        seed=42, 
        dim_limit=1024, 
        debug=False
    ):
        self.max_size = max_size
        self.f = 8
        self.max_tokens = (max_size[0]/self.f) * (max_size[1]/self.f)
        self.div = divisible
        self.min_dim = min_dim
        self.dim_limit = dim_limit
        self.base_res = base_res
        self.debug = debug
        self.prng = get_prng(seed)
        self.gen_buckets()

    def gen_buckets(self):
        if self.debug:
            timer = time.perf_counter()

        resolutions = []
        aspects = []

        # Generate resolution buckets, first increasing width then height
        w = self.min_dim
        while (w/self.f) * (self.min_dim/self.f) <= self.max_tokens and w <= self.dim_limit:
            h = self.min_dim
            got_base = False
            while (w/self.f) * ((h+self.div)/self.f) <= self.max_tokens and (h+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                h += self.div
            if (w != self.base_res[0] or h != self.base_res[1]) and got_base:
                resolutions.append(self.base_res)
                aspects.append(1)
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            w += self.div

        # Now increasing height then width
        h = self.min_dim
        while (h/self.f) * (self.min_dim/self.f) <= self.max_tokens and h <= self.dim_limit:
            w = self.min_dim
            got_base = False
            while (h/self.f) * ((w+self.div)/self.f) <= self.max_tokens and (w+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                w += self.div
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            h += self.div

        # De-duplicate, sort
        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]
        self.resolutions = sorted(res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        self.aspects = np.array([res_map[x] for x in self.resolutions])
        self.resolutions = np.array(self.resolutions)

        if self.debug:
            timer = time.perf_counter() - timer
            print(f"resolutions:\n{self.resolutions}")
            print(f"aspects:\n{self.aspects}")
            print(f"gen_buckets: {timer:.5f}s")

    def get_ideal_resolution(self, image_size) -> tuple[int, int]:
        """Given the (width, height) of an image, find the best (bucketed) resolution."""
        w, h = image_size
        aspect = float(w)/float(h)
        # We pick the resolution whose log(aspect) is closest to log(aspect of image)
        bucket_id = np.abs(np.log(self.aspects) - np.log(aspect)).argmin()
        return tuple(self.resolutions[bucket_id])

def get_processed_files(tracking_file):
    processed_files = set()
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            processed_files = set(line.strip() for line in f)
    return processed_files

def add_processed_file(tracking_file, filename):
    with open(tracking_file, 'a') as f:
        f.write(f"{filename}\n")

# --------------------------------------------------------------------------------
# ------------------------- Image Download/Upload ---------------------------------
# --------------------------------------------------------------------------------

def download_with_retry(repo_id, filename, repo_type, local_dir, max_retries=500, retry_delay=60):
    """
    Tries to download 'filename' from a Hugging Face repo up to `max_retries` times.
    """
    api = HfApi()
    for attempt in range(max_retries):
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_dir=local_dir
            )
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Download failed: {e}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Skipping file: {filename}")
                return None

def upload_with_retry(file_path, name_in_repo, max_retries=500, retry_delay=60):
    """
    Tries to upload 'file_path' to a Hugging Face dataset repo up to `max_retries` times.
    """
    api = HfApi()
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=name_in_repo,
                repo_id=f"{USERNAME}/preprocessed_{DATASET}_DCAE",
                repo_type="dataset",
            )
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Upload failed: {str(e)}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Failed to upload: {file_path}")
                return False

# --------------------------------------------------------------------------------
# ------------------------- Image Preprocessing -----------------------------------
# --------------------------------------------------------------------------------

def resize_and_crop(image, target_size, rng=None):
    """
    Resizes and (optionally) random-crops an image to match the target_size (width, height).
    If aspect ratios match, it just does a single BICUBIC resize.
    Otherwise, it resizes in one dimension and random-crops the other dimension.
    """
    if rng is None:
        rng = np.random

    target_w, target_h = target_size
    aspect_ratio = image.width / image.height
    target_aspect_ratio = target_w / target_h
    
    if abs(aspect_ratio - target_aspect_ratio) < 1e-6:
        # Aspect ratios match, resize directly
        image = image.resize((target_w, target_h), Image.BICUBIC)
    else:
        # Resize while preserving aspect ratio, then random crop
        if aspect_ratio > target_aspect_ratio:
            # Image is wider than target
            new_height = target_h
            new_width = int(aspect_ratio * new_height)
        else:
            # Image is taller or equal
            new_width = target_w
            new_height = int(new_width / aspect_ratio)
        image = image.resize((new_width, new_height), Image.BICUBIC)
        # Random crop to target size
        left = rng.randint(0, new_width - target_w + 1)
        upper = rng.randint(0, new_height - target_h + 1)
        image = image.crop((left, upper, left + target_w, upper + target_h))
    return image

def preprocess_image(image):
    """
    Converts a PIL image into a torch.Tensor in [-1, 1].
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  
    ])
    return transform(image)

def single_image_process(image_bytes, new_resolution, rng=None):
    """
    Convert bytes -> PIL -> resize & crop -> to tensor -> normalize.
    """
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = resize_and_crop(image, new_resolution, rng=rng)
    return preprocess_image(image)

def process_images_threaded(batch, image_col, new_resolution, rng=None, max_workers=16):
    """
    Process a list of image bytes in parallel using threads.
    If you find even threading overhead is too high, you can remove it 
    and do it in a simple for-loop.
    """
    image_bytes_list = [batch[image_col][i].as_py() for i in range(len(batch))]
    if rng is None:
        rng = np.random

    # We create a local PRNG seed per thread in a closure if desired
    def _process(x):
        return single_image_process(x, new_resolution, rng)

    # Threaded approach (CPU-bound ops)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        processed_list = list(executor.map(_process, image_bytes_list))

    # Combine into single tensor
    return torch.stack(processed_list, dim=0)

# --------------------------------------------------------------------------------
# -------------------------- Model Wrappers ---------------------------------------
# --------------------------------------------------------------------------------

class TextEmbedder:
    """
    Generic class to handle text embeddings. 
    `cls=True` means we want BERT-style [CLS] usage.
    """
    def __init__(self, model, tokenizer, device, cls=False, max_length=64):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cls = cls
        self.max_length = max_length

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(self, captions):
        inputs_list = []
        unpadded_lens = []

        for prompt in captions:
            if self.cls:
                text = "[CLS]" + prompt
            else:
                text = prompt

            # If your tokenizer needs special handling, do so here. 
            # For demonstration, we'll do a naive .encode()
            input_ids = self.tokenizer.encode(
                text, 
                return_tensors="pt", 
                padding=False
            ).to(self.device).squeeze()

            # Keep track of length
            unpadded_len = input_ids.shape[-1] - (1 if self.cls else 0)
            unpadded_lens.append(unpadded_len)

            # If shorter than max_length, pad up to max_length
            # If longer, truncate
            needed_len = self.max_length + (1 if self.cls else 0)
            if input_ids.shape[0] < needed_len:
                pad_size = needed_len - input_ids.shape[0]
                padding = torch.ones((pad_size,), dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat([input_ids, padding])
            else:
                input_ids = input_ids[:needed_len]

            inputs_list.append(input_ids)

        inputs = torch.stack(inputs_list, dim=0)  # (bs, max_length)

        outputs = self.model(inputs, output_hidden_states=True)
        if self.cls:
            # BERT style: 
            # last hidden state shape: (bs, seq_len, hidden_dim)
            # We also might have a pooler_output
            hidden = outputs.hidden_states[-1]  # (bs, seq_len, dim)
            # separate [CLS] token 
            vec = hidden[:, 0, :]        # (bs, dim)
            embeddings = hidden[:, 1:, ] # (bs, seq_len-1, dim)
        else:
            hidden = outputs.hidden_states[-1]
            vec = outputs.pooler_output  # or some final pooling
            embeddings = hidden

        return embeddings, vec, unpadded_lens

class ImageEmbedder:
    """
    Handles VAE latent encoding via diffusers' Autoencoder.
    """
    def __init__(self, ae_model):
        self.model = ae_model

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(self, images):
        latents = self.model.encode(images).latent
        return latents

# --------------------------------------------------------------------------------
# --------------------- Main Processing Logic (Single GPU) -----------------------
# --------------------------------------------------------------------------------

def check_and_upload_unuploaded_files():
    """
    At startup, look for any local preprocessed parquet files 
    that may not yet be uploaded. If found, upload them.
    """
    if not UPLOAD_TO_HUGGINGFACE:
        return

    preprocessed_dir = os.path.join(DATASET_DIR_BASE, f"preprocessed_{DATASET}")
    if not os.path.exists(preprocessed_dir):
        return

    # Gather local parquet files
    existing_local_parquets = []
    for root, dirs, files in os.walk(preprocessed_dir):
        for file in files:
            if file.endswith(".parquet"):
                filepath = os.path.join(root, file)
                existing_local_parquets.append(filepath)

    fs = HfFileSystem()
    on_hub = fs.glob(f"datasets/{USERNAME}/preprocessed_{DATASET}_DCAE/**/*.parquet")
    on_hub_basenames = set(os.path.basename(x) for x in on_hub)

    # Upload anything local that's missing on the hub
    for local_path in existing_local_parquets:
        fname = os.path.basename(local_path)
        if fname not in on_hub_basenames:
            # We'll define the path structure in the repo 
            # by everything after "preprocessed_{DATASET}/"
            rel_path = local_path.split(f"preprocessed_{DATASET}/")[-1]
            success = upload_with_retry(local_path, rel_path)
            if success and DELETE_AFTER_PROCESSING:
                os.remove(local_path)


def process_parquet_file(
    parquet_filepath, 
    device, 
    bucket_manager, 
    ae, 
    siglip, 
    bert, 
    captions_json, 
    tracking_file
):
    """
    Processes a single parquet file:
    - reads it
    - picks a bucket resolution
    - runs image->latent, text->embedding
    - writes out new parquet
    - optionally uploads + cleans up
    """

    # Skip if already processed
    basename = os.path.basename(parquet_filepath)
    if basename in get_processed_files(tracking_file):
        return  # already processed

    # Read
    df = pq.read_table(parquet_filepath)
    if len(df) == 0:
        print(f"Parquet file {parquet_filepath} is empty!")
        return

    # Pick a resolution once for the entire parquet
    # because they share the same aspect ratio
    first_img_bytes = df[IMAGE_COLUMN_NAME][0].as_py()
    sample_img = Image.open(io.BytesIO(first_img_bytes))
    original_resolution = sample_img.size  # (w, h)

    new_resolution = bucket_manager.get_ideal_resolution(original_resolution)

    new_rows = []
    start_time = time.time()
    total_images_processed = 0

    # Process in mini-batches
    for batch_start in tqdm(range(0, len(df), BS), "Processing parquet"):
        batch = df.slice(batch_start, BS)

        # Preprocess images (threaded to speed up CPU resizing)
        image_tensors = process_images_threaded(
            batch, 
            IMAGE_COLUMN_NAME, 
            new_resolution, 
            rng=np.random,    # or a stable RNG 
            max_workers=8
        ).to(device)
    
        # Extract captions
        captions = [
            captions_json[batch[IMAGE_ID_COLUMN_NAME][i].as_py()] 
            for i in range(len(batch))
        ]

        # Encode images => latents
        latents = ae.generate(image_tensors)

        # SigLip embeddings
        sig_emb, sig_vec, sig_unpad = siglip.generate(captions)

        # BERT embeddings
        bert_emb, bert_vec, bert_unpad = bert.generate(captions)

        # Collect output rows
        for i in range(len(batch)):
            new_row = {
                'image_id': batch[IMAGE_ID_COLUMN_NAME][i].as_py(),
                'caption': captions[i],
                'ae_latent': latents[i].cpu().to(torch.float16).numpy().flatten(),
                'ae_latent_shape': latents[i].shape,
                'siglip_emb': sig_emb[i].cpu().to(torch.float16).numpy().flatten(),
                'siglip_vec': sig_vec[i].cpu().to(torch.float16).numpy().flatten(),
                'siglip_unpadded_len': sig_unpad[i],
                'bert_emb': bert_emb[i].cpu().to(torch.float16).numpy().flatten(),
                'bert_vec': bert_vec[i].cpu().to(torch.float16).numpy().flatten(),
                'bert_unpadded_len': bert_unpad[i],
            }
            new_rows.append(new_row)

        total_images_processed += len(batch)

    elapsed = time.time() - start_time
    img_per_sec = total_images_processed / elapsed if elapsed > 0 else 0.0
    print(f"Processed {total_images_processed} images from {basename} in {elapsed:.2f}s ({img_per_sec:.2f} img/s).")

    # Write new parquet
    new_table = pa.Table.from_pylist(new_rows)
    new_parquet_dir = os.path.join(DATASET_DIR_BASE, f"preprocessed_{DATASET}", f"{new_resolution[1]}x{new_resolution[0]}")
    os.makedirs(new_parquet_dir, exist_ok=True)
    new_parquet_path = os.path.join(new_parquet_dir, basename)
    pq.write_table(new_table, new_parquet_path)

    # Upload
    if UPLOAD_TO_HUGGINGFACE:
        rel_path_in_repo = f"{new_resolution[1]}x{new_resolution[0]}/{basename}"
        success = upload_with_retry(new_parquet_path, rel_path_in_repo)
        if success and DELETE_AFTER_PROCESSING:
            # If user wants to remove local file after successful upload
            os.remove(new_parquet_path)

    # Mark original as processed
    add_processed_file(tracking_file, basename)

    # Delete original parquet if desired
    if DELETE_AFTER_PROCESSING and os.path.exists(parquet_filepath):
        os.remove(parquet_filepath)


def process_dataset():
    """
    Main single-process, single-GPU pipeline:
      1. Builds model + tokenizers on GPU: AE, SigLip, BERT
      2. Gathers all unprocessed parquet files from the HF dataset
      3. Downloads them, processes them, and uploads the result
    """
    # ---- Step 1: If requested, see if there's any local preprocessed files 
    #              that haven't been uploaded yet
    check_and_upload_unuploaded_files()

    # ---- Step 2: Prepare device + models
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Autoencoder
    ae_model = AutoencoderDC.from_pretrained(
        f"mit-han-lab/{AE_HF_NAME}",
        torch_dtype=torch.bfloat16,
        cache_dir=f"{MODELS_DIR_BASE}/{AE_HF_NAME}"
    ).to(device)
    ae = ImageEmbedder(ae_model)

    # Siglip
    siglip_model = SiglipTextModel.from_pretrained(
        SIGLIP_HF_NAME, 
        cache_dir=f"{MODELS_DIR_BASE}/siglip"
    ).to(device)
    siglip_tokenizer = SiglipTokenizer.from_pretrained(
        SIGLIP_HF_NAME, 
        cache_dir=f"{MODELS_DIR_BASE}/siglip"
    )
    siglip = TextEmbedder(siglip_model, siglip_tokenizer, device, cls=False)

    # Modern BERT
    bert_model = ModernBertModel.from_pretrained(
        BERT_HF_NAME, 
        cache_dir=f"{MODELS_DIR_BASE}/modernbert"
    ).to(device)
    bert_tokenizer = AutoTokenizer.from_pretrained(
        BERT_HF_NAME, 
        cache_dir=f"{MODELS_DIR_BASE}/modernbert"
    )
    bert = TextEmbedder(bert_model, bert_tokenizer, device, cls=True)

    # Load captions JSON
    captions_path = hf_hub_download(
        repo_id="SwayStar123/preprocessed_commoncatalog-cc-by", 
        filename="prompts.json", 
        repo_type="dataset", 
        local_dir=f"{DATASET_DIR_BASE}/{DATASET}_captions"
    )
    with open(captions_path, "r") as f:
        captions_json = json.load(f)

    # Bucket manager for picking resolutions
    bucket_manager = BucketManager()

    # Tracking file (to mark original parquets as processed)
    tracking_file = os.path.join(DATASET_DIR_BASE, f"{DATASET}_processed_files.txt")

    # ---- Step 3: Gather remote parquet file paths and filter out processed
    fs = HfFileSystem()
    parquet_files = []
    for partition in PARTITIONS:
        # e.g. "datasets/common-canvas/commoncatalog-cc-by/0/**/*.parquet"
        partition_glob = f"datasets/common-canvas/{DATASET}/{partition}/**/*.parquet"
        found = fs.glob(partition_glob)
        parquet_files.extend(found)

    already_processed = get_processed_files(tracking_file)
    processed_on_hub = fs.glob(f"datasets/{USERNAME}/preprocessed_{DATASET}_DCAE/**/*.parquet")
    processed_on_hub_basenames = set(os.path.basename(x) for x in processed_on_hub)

    # Filter
    def is_unprocessed(p):
        basename = os.path.basename(p)
        return (basename not in already_processed) and (basename not in processed_on_hub_basenames)

    unprocessed_files = [pf for pf in parquet_files if is_unprocessed(pf)]
    print(f"Found {len(unprocessed_files)} unprocessed parquet files.")

    # ---- Step 4: Loop over unprocessed files, download, process, upload
    for parquet_path in tqdm(unprocessed_files, desc="Overall Progress"):
        # Download to local
        rel_path = parquet_path.split(f"{DATASET}/")[-1]
        local_file = download_with_retry(
            repo_id=f"common-canvas/{DATASET}",
            filename=rel_path,
            repo_type="dataset",
            local_dir=f"{DATASET_DIR_BASE}/{DATASET}"
        )
        if not local_file:
            # Skip if failed to download
            continue

        # Process
        process_parquet_file(
            parquet_filepath=local_file,
            device=device,
            bucket_manager=bucket_manager,
            ae=ae,
            siglip=siglip,
            bert=bert,
            captions_json=captions_json,
            tracking_file=tracking_file
        )

    print("All done!")


if __name__ == "__main__":
    process_dataset()
