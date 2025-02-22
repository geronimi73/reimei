import torch
from einops import rearrange, repeat
from torch import Tensor

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, dropout: float = 0.0) -> Tensor:
    if pe is not None:
        q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

def rope_ids(bs, h, w, seq_len, device, dtype):
    img_ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h, device=device, dtype=dtype)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w, device=device, dtype=dtype)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    txt_ids = torch.zeros(bs, seq_len, 3, device=device, dtype=dtype)

    return txt_ids, img_ids

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)