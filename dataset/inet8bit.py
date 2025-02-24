class ImageNetDataset(Dataset):
    def __init__(self, data_path, labels_path=None):
        self.data = np.memmap(data_path, dtype='uint8', mode='r', shape=(1281152, 4096))
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label, label_text = self.labels[idx]
        image = image.astype(np.float32).reshape(4, 32, 32)
        image = (image / 255.0 - 0.5) * 24.0
        return image, label, label_text
    
class InfiniteDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.siglip_model = SiglipTextModel.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip").to(device, DTYPE)
        self.siglip_tokenizer = SiglipTokenizer.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip")
        self.device = "cuda"

    def __iter__(self):
        while True:
            for images, labels, ltxt in self.dataset:
                captions = [ltxt[i] for i in range(len(ltxt))]
                siglip_emb, siglip_vec = self.encode(captions)
                batch = {
                    'ae_latent': images,
                    'siglip_emb': siglip_emb,
                    'siglip_vec': siglip_vec,
                    'bert_emb': torch.zeros(len(images), 1, BERT_EMBED_DIM).to(device, dtype=DTYPE),
                    'bert_vec': torch.zeros(len(images), BERT_EMBED_DIM).to(device, dtype=DTYPE),
                }
                yield batch

    @torch.no_grad
    def encode(self, captions):
        bs = len(captions)

        # Encode the captions
        s_tokens = self.siglip_tokenizer(captions, padding='longest', truncation=True, return_tensors="pt").to(self.device)
        siglip_outputs = self.siglip_model(**s_tokens, output_hidden_states=True)
        siglip_embedding = siglip_outputs.hidden_states[-1]
        siglip_vec = siglip_outputs.pooler_output

        return siglip_embedding, siglip_vec