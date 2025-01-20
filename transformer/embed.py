import torch
import torch.nn as nn
import math

def rope_1d(dim: int, seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """
    Returns a [seq_len, dim] 1D rotary embedding (cos/sin interleaved).
    
    - We treat each pair of channels as (cos, sin).
    - If dim=8, then we have dim//2=4 distinct frequencies, each used for 1 cos+sin pair.
    """
    assert dim % 2 == 0, f"rope_1d requires an even dim, got dim={dim}"
    half_dim = dim // 2

    # positions: shape [seq_len]
    positions = torch.arange(seq_len, dtype=torch.float)

    # freq for each pair
    freq_seq = torch.arange(half_dim, dtype=torch.float) / float(half_dim)
    freq_seq = base ** (-2.0 * freq_seq)  # shape [half_dim]

    # Outer product => [seq_len, half_dim]
    phase = positions[:, None] * freq_seq[None, :]  # [seq_len, half_dim]

    # Cos, sin => shape [seq_len, half_dim] each
    cos_ = torch.cos(phase)
    sin_ = torch.sin(phase)

    # Interleave cos,sin into final [seq_len, dim]
    emb = torch.stack([cos_, sin_], dim=-1)  # [seq_len, half_dim, 2]
    emb = emb.view(seq_len, dim)            # [seq_len, dim]
    return emb

def rope_2d(dim: int, height: int, width: int, base: float = 10000.0) -> torch.Tensor:
    """
    Returns a [height*width, dim] 2D rotary embedding.
    - The first (dim/2) covers the row dimension, the second (dim/2) covers the col dimension.
    - Each half is cos/sin interleaved as in rope_1d.
    """
    assert dim % 2 == 0, "rope_2d requires an even dim"
    half = dim // 2

    # 1) row part => [height, half]
    # We can reuse rope_1d logic: rope_1d(half, height)
    row_emb_1d = rope_1d(half, height, base=base)  # [height, half]
    # 2) col part => [width, half]
    col_emb_1d = rope_1d(half, width, base=base)   # [width, half]

    # We want a [height, width, dim] => row_emb in first half, col_emb in second half
    # row_emb_1d => shape [height, half] -> broadcast across width
    # col_emb_1d => shape [width, half] -> broadcast across height
    row_emb_2d = row_emb_1d.unsqueeze(1).expand(height, width, half)  # [H,W,half]
    col_emb_2d = col_emb_1d.unsqueeze(0).expand(height, width, half)  # [H,W,half]

    # combine => [H,W,dim]
    emb_2d = torch.cat([row_emb_2d, col_emb_2d], dim=-1)  # [H,W, dim]
    emb_2d = emb_2d.view(height*width, dim)               # flatten to [N, dim]
    return emb_2d

def sincos_2d(embed_dim, h, w):
    """
    :param embed_dim: dimension of the embedding
    :param h: height of the grid
    :param w: width of the grid
    :return: [h*w, embed_dim] or [1+h*w, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, h, w])
    pos_embed = sincos_2d_from_grid(embed_dim, grid)
    return pos_embed

def sincos_2d_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = sincos_1d(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = sincos_1d(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
    return emb

def sincos_1d(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = MLPEmbedder(frequency_embedding_size, hidden_size)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D torch.Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) torch.Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        # if inference with fp16, embedding.half()
        return embedding 

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size) 
        t_emb = self.mlp(t_freq)#.half())
        return t_emb
    
class OutputLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.mlp(x)
        return x