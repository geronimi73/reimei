import json
import os
from transformers import SiglipTokenizer, SiglipTextModel, AutoTokenizer, ModernBertModel
from diffusers import AutoencoderDC
import torch
import time
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.multiprocessing import Queue, Process, Value, Event, set_start_method
from tqdm import tqdm
from huggingface_hub import HfFileSystem, hf_hub_download, HfApi
import threading
import io
import concurrent.futures

DATASET = "commoncatalog-cc-by"
DATASET_DIR_BASE = "../datasets"
MODELS_DIR_BASE = "../models"
AE_HF_NAME = "dc-ae-f32c32-mix-1.0-diffusers"
SIGLIP_HF_NAME = "google/siglip-so400m-patch14-384"
BERT_HF_NAME = "answerdotai/ModernBERT-large"
IMAGE_COLUMN_NAME = "jpg"
IMAGE_ID_COLUMN_NAME = "key"
BS = 32
# Deletes original parquet files after processing
DELETE_AFTER_PROCESSING = True
UPLOAD_TO_HUGGINGFACE = True
USERNAME = "SwayStar123"
PARTITIONS = [0,1,2,3,4,5,6,7,8,9]

def get_prng(seed):
    return np.random.RandomState(seed)

class BucketManager:
    def __init__(self, max_size=(512,512), divisible=32, min_dim=256, base_res=(512,612), bsz=64, world_size=1, global_rank=0, max_ar_error=4, seed=42, dim_limit=1024, debug=False):
        self.max_size = max_size
        self.f = 8
        self.max_tokens = (max_size[0]/self.f) * (max_size[1]/self.f)
        self.div = divisible
        self.min_dim = min_dim
        self.dim_limit = dim_limit
        self.base_res = base_res
        self.bsz = bsz
        self.world_size = world_size
        self.global_rank = global_rank
        self.max_ar_error = max_ar_error
        self.prng = get_prng(seed)
        epoch_seed = self.prng.tomaxint() % (2**32-1)
        self.epoch_prng = get_prng(epoch_seed) # separate prng for sharding use for increased thread resilience
        self.epoch = None
        self.left_over = None
        self.batch_total = None
        self.batch_delivered = None

        self.debug = debug

        self.gen_buckets()

    def gen_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        resolutions = []
        aspects = []
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
        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]
        self.resolutions = sorted(res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        self.aspects = np.array(list(map(lambda x: res_map[x], self.resolutions)))
        self.resolutions = np.array(self.resolutions)
        if self.debug:
            timer = time.perf_counter() - timer
            print(f"resolutions:\n{self.resolutions}")
            print(f"aspects:\n{self.aspects}")
            print(f"gen_buckets: {timer:.5f}s")

    def get_ideal_resolution(self, image_size) -> tuple[int, int]:
        w, h = image_size
        aspect = float(w)/float(h)
        bucket_id = np.abs(np.log(self.aspects) - np.log(aspect)).argmin()
        return self.resolutions[bucket_id]

def bytes_to_pil_image(image_bytes):
    return Image.open(io.BytesIO(image_bytes))

def resize_and_crop(image, target_size):
    # image: PIL Image
    # target_size: (width, height)
    target_w, target_h = target_size
    aspect_ratio = image.width / image.height
    target_aspect_ratio = target_w / target_h
    if abs(aspect_ratio - target_aspect_ratio) < 1e-6:
        # Aspect ratios match, resize directly
        image = image.resize((target_w, target_h), Image.BICUBIC)
    else:
        # Resize while preserving aspect ratio, then random crop
        if aspect_ratio > target_aspect_ratio:
            # Image is wider than target, resize height to target height
            new_height = target_h
            new_width = int(aspect_ratio * new_height)
        else:
            # Image is taller than target, resize width to target width
            new_width = target_w
            new_height = int(new_width / aspect_ratio)
        image = image.resize((new_width, new_height), Image.BICUBIC)
        # Random crop to target size
        left = np.random.randint(0, new_width - target_w + 1)
        upper = np.random.randint(0, new_height - target_h + 1)
        image = image.crop((left, upper, left + target_w, upper + target_h))
    return image

def preprocess_image(image):
    # Ensure image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Convert to tensor, normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0,1], shape (C,H,W)
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1,1]
    ])
    image_tensor = transform(image)
    return image_tensor

class TextEmbedder:
    def __init__(self, model, tokenizer, device, cls=False, max_length=64):
        """Initialize with the model, tokenizer, and device."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cls = cls
        self.max_length = max_length

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(self, captions):
        """Generate embeddings for a list of captions."""
        inputs_list = []
        unpadded_lens = []

        for prompt in captions:
            # If BERT-style, prepend [CLS]
            if self.cls:
                text = "[CLS]" + prompt
            else:
                text = prompt
            
            input = self.tokenizer.encode(text, return_tensors="pt", padding=True).to(self.device).squeeze()
            unpadded_lens.append(input.shape[-1] - (1 if self.cls else 0))
            if input.shape[0] < self.max_length + (1 if self.cls else 0):
                padding = torch.ones((self.max_length + (1 if self.cls else 0) - input.shape[0]), dtype=input.dtype, device=input.device)
                input = torch.cat([input, padding])
            else:
                input = input[:self.max_length + (1 if self.cls else 0)]
            inputs_list.append(input)

        inputs = torch.stack(inputs_list)
        if self.cls:
            outputs = self.model(inputs, output_hidden_states=True).hidden_states[-1] # (bs, 65, 768). The 65 is CLS + 64 tokens. So we need to seperate the CLS token from the rest.
            embeddings = outputs[:, 1:, :] # (bs, 64, 1024)
            vec = outputs[:, 0, :] # (bs, 1024)
        else:
            outputs = self.model(inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]
            vec = outputs.pooler_output

        return embeddings, vec, unpadded_lens

class ImageEmbedder:
    def __init__(self, model):
        """Initialize with the model."""
        self.model = model

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(self, images):
        """Generate latents for a set of images."""
        latents = self.model.encode(images).latent
        return latents

def get_processed_files(tracking_file):
    processed_files = set()
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            processed_files = set(line.strip() for line in f)
    return processed_files

def add_processed_file(tracking_file, filename):
    with open(tracking_file, 'a') as f:
        f.write(f"{filename}\n")

def download_with_retry(repo_id, filename, repo_type, local_dir, max_retries=500, retry_delay=60):
    for attempt in range(max_retries):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=local_dir)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Download failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Skipping file: {filename}")
                return None

def download_and_queue_parquets(queue, download_progress, total_files, download_complete_event, tracking_file, pause_event):
    fs = HfFileSystem()
    parquet_files = []
    for partition in PARTITIONS:
        p_f = fs.glob(f"datasets/common-canvas/{DATASET}/{str(partition)}/**/*.parquet")
        parquet_files.extend(p_f)

    processed_files = get_processed_files(tracking_file)
    processed_files_on_hub = fs.glob(f"datasets/{USERNAME}/preprocessed_{DATASET}_DCAE/**/*.parquet")
    processed_files_on_hub = set(os.path.basename(file) for file in processed_files_on_hub)

    unprocessed_files = [file for file in parquet_files if os.path.basename(file) not in processed_files and os.path.basename(file) not in processed_files_on_hub]

    total_files.value = len(unprocessed_files)
    
    for file in unprocessed_files:
        # Check if download should be paused
        pause_event.wait()
        
        local_file = download_with_retry(
            repo_id=f"common-canvas/{DATASET}",
            filename=file.split(f"{DATASET}/")[-1],
            repo_type="dataset",
            local_dir=f"{DATASET_DIR_BASE}/{DATASET}"
        )
        
        if local_file:
            queue.put(local_file)
            with download_progress.get_lock():
                download_progress.value += 1
    
    download_complete_event.set()

def upload_with_retry(file_path, name_in_repo, max_retries=500, retry_delay=60):
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

def upload_worker(upload_queue, upload_complete_event):
    while not (upload_complete_event.is_set() and upload_queue.empty()):
        try:
            file_path, name_in_repo = upload_queue.get(timeout=10)
        except:
            continue

        if upload_with_retry(file_path, name_in_repo):
            print(f"Successfully uploaded: {file_path}")
            if DELETE_AFTER_PROCESSING:
                os.remove(file_path)
        else:
            print(f"Failed to upload: {file_path}")

def single_image_process(image_bytes, new_resolution):
    """Convert bytes -> PIL -> resize & crop -> to tensor -> normalize."""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = resize_and_crop(image, new_resolution)  # your existing function
    return preprocess_image(image)                  # your existing function

def process_images_threaded(batch, image_col, new_resolution, max_workers=16):
    """Process a list of image bytes in parallel using threads."""
    # Extract bytes from the parquet batch
    image_bytes_list = [batch[image_col][i].as_py() for i in range(len(batch))]

    # Run parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        processed_list = list(
            executor.map(
                lambda x: single_image_process(x, new_resolution),
                image_bytes_list
            )
        )

    # Stack into a single tensor
    images_tensor = torch.stack(processed_list, dim=0)
    return images_tensor


def process_parquets(rank, world_size, queue, process_progress, total_files, total_images, download_complete_event, tracking_file, upload_queue):
    # Move CUDA initialization inside this function
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    ae = AutoencoderDC.from_pretrained(f"mit-han-lab/{AE_HF_NAME}", torch_dtype=torch.bfloat16, cache_dir=f"{MODELS_DIR_BASE}/{AE_HF_NAME}").to(device)
    ae = ImageEmbedder(ae)

    siglip_model = SiglipTextModel.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip").to(device)
    siglip_tokenizer = SiglipTokenizer.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip")
    siglip = TextEmbedder(siglip_model, siglip_tokenizer, device, cls=False)

    bert_model = ModernBertModel.from_pretrained(BERT_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/modernbert").to(device)
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/modernbert")
    bert = TextEmbedder(bert_model, bert_tokenizer, device, cls=True)

    captions_json = hf_hub_download("SwayStar123/preprocessed_commoncatalog-cc-by", filename="prompts.json", repo_type="dataset", local_dir=f"commoncatalog-cc-by_captions")
    captions_json = json.load(open(captions_json))

    bucket_manager = BucketManager()
    
    while not (download_complete_event.is_set() and queue.empty()):
        try:
            parquet_filepath = queue.get(timeout=10)
        except:
            continue

        if os.path.basename(parquet_filepath) in get_processed_files(tracking_file):
            continue

        # Load parquet file
        df = pq.read_table(parquet_filepath)
        
        # Get ideal resolution
        sample_image_bytes = df[IMAGE_COLUMN_NAME][0].as_py()
        sample_image = bytes_to_pil_image(sample_image_bytes)
        original_resolution = sample_image.size
        new_resolution = bucket_manager.get_ideal_resolution(original_resolution)
        
        new_rows = []
        
        for batch_start in tqdm(range(0, len(df), BS), desc=f"Processing {os.path.basename(parquet_filepath)}", position=3):
            batch = df.slice(batch_start, BS)
            
            # Resize images
            # images = [bytes_to_pil_image(img.as_py()) for img in batch[IMAGE_COLUMN_NAME]]
            # resized_images = [resize_and_crop(img, new_resolution) for img in images]
            # image_tensors = torch.stack([preprocess_image(img) for img in resized_images]).to(device)

            image_tensors = process_images_threaded(batch, IMAGE_COLUMN_NAME, new_resolution)
            image_tensors = image_tensors.to(device)

            captions = [captions_json[batch[IMAGE_ID_COLUMN_NAME][i].as_py()] for i in range(len(batch))]

            # Generate VAE latents
            latents = ae.generate(image_tensors)
            
            # Generate text embeddings
            sig_emb, sig_vec, sig_unpad = siglip.generate(captions)
            bert_emb, bert_vec, bert_unpad = bert.generate(captions)
            
            # Add only processed outputs to new rows
            for i in range(len(batch)):
                new_row = {
                    'image_id': batch[IMAGE_ID_COLUMN_NAME][i].as_py(),
                    'caption': captions[i],
                    'ae_latent': latents[i].cpu().to(torch.float16).numpy().flatten(),
                    'ae_latent_shape': latents[i].shape,  # Store shape separately
                    'siglip_emb': sig_emb[i].cpu().to(torch.float16).numpy().flatten(),
                    'siglip_vec': sig_vec[i].cpu().to(torch.float16).numpy().flatten(),
                    'siglip_unpadded_len': sig_unpad[i],
                    'bert_emb': bert_emb[i].cpu().to(torch.float16).numpy().flatten(),
                    'bert_vec': bert_vec[i].cpu().to(torch.float16).numpy().flatten(),
                    'bert_unpadded_len': bert_unpad[i],
                }
                new_rows.append(new_row)
            
            # Update total images processed
            with total_images.get_lock():
                total_images.value += len(batch)
        
        # Create new parquet file
        new_df = pa.Table.from_pylist(new_rows)
        new_parquet_dir = os.path.join(DATASET_DIR_BASE, f"preprocessed_{DATASET}", f"{new_resolution[1]}x{new_resolution[0]}")
        os.makedirs(new_parquet_dir, exist_ok=True)
        new_parquet_path = os.path.join(new_parquet_dir, os.path.basename(parquet_filepath))
        pq.write_table(new_df, new_parquet_path)

        if UPLOAD_TO_HUGGINGFACE:
            name_in_repo = f"{new_resolution[1]}x{new_resolution[0]}/{os.path.basename(parquet_filepath)}"
            upload_queue.put((new_parquet_path, name_in_repo))

        add_processed_file(tracking_file, os.path.basename(parquet_filepath))

        # Delete original parquet file if DELETE_AFTER_PROCESSING is True
        if DELETE_AFTER_PROCESSING:
            os.remove(parquet_filepath)

        # Update process progress
        with process_progress.get_lock():
            process_progress.value += 1

def process_dataset():
    # Set start method to 'spawn'
    set_start_method('spawn', force=True)

    world_size = torch.cuda.device_count()
    queue = Queue()
    upload_queue = Queue()
    download_progress = Value('i', 0)
    process_progress = Value('i', 0)
    total_files = Value('i', 0)
    total_images = Value('i', 0)
    download_complete_event = Event()
    upload_complete_event = Event()
    pause_event = Event()
    pause_event.set()  # Start in a non-paused state
    
    # Create tracking file path
    tracking_file = os.path.join(DATASET_DIR_BASE, f"{DATASET}_processed_files.txt")
    
    # Start the download thread
    download_thread = threading.Thread(target=download_and_queue_parquets, args=(queue, download_progress, total_files, download_complete_event, tracking_file, pause_event))
    download_thread.start()

    # Start the upload thread if enabled
    if UPLOAD_TO_HUGGINGFACE:
        upload_thread = threading.Thread(target=upload_worker, args=(upload_queue, upload_complete_event))
        upload_thread.start()
    
    processes = []
    for rank in range(world_size):
        p = Process(target=process_parquets, args=(rank, world_size, queue, process_progress, total_files, total_images, download_complete_event, tracking_file, upload_queue))
        p.start()
        processes.append(p)
    
    # Progress bars with img/s estimate
    start_time = time.time()
    with tqdm(total=None, desc="Downloading files", position=1) as download_pbar, \
         tqdm(total=None, desc="Processing files", position=2) as process_pbar:
        last_images = 0
        while not (download_complete_event.is_set() and queue.empty()):
            current_download_progress = download_progress.value
            current_process_progress = process_progress.value
            current_images = total_images.value
            elapsed_time = time.time() - start_time
            
            # Update totals if changed
            if download_pbar.total != total_files.value:
                download_pbar.total = total_files.value
                process_pbar.total = total_files.value
            
            # Update progress bars
            download_pbar.n = current_download_progress
            process_pbar.n = current_process_progress
            
            # Calculate images per second
            if elapsed_time > 0:
                img_per_second = current_images / elapsed_time
                new_images = current_images - last_images
                if new_images > 0:
                    img_per_second = (img_per_second + new_images) / 2  # Average of overall and recent speed
                
                process_pbar.set_postfix({'img/s': f'{img_per_second:.2f}'})
            
            # Pause or resume download based on queue size
            if queue.qsize() > world_size * 2:
                pause_event.clear()  # Pause download
            else:
                pause_event.set()  # Resume download
            
            download_pbar.refresh()
            process_pbar.refresh()
            
            last_images = current_images
            time.sleep(1)
    
    download_thread.join()
    for p in processes:
        p.join()

    if UPLOAD_TO_HUGGINGFACE:
        upload_complete_event.set()
        upload_thread.join()

if __name__ == "__main__":
    process_dataset()