from dataclasses import dataclass
from einops import rearrange
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import RMSNorm

from transformer.math import attention
from transformer.moe import EC_SparseMoeBlock, TC_SparseMoeBlock

class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


#################################################################################
#                                 Core DiT Modules                              #
#################################################################################
from timm.models.vision_transformer import Attention

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self, hidden_size, num_heads, mlp_dim,
        num_experts=8, num_experts_per_tok=2, pretraining_tp=2, num_shared_experts=2, use_expert_choice=False, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0) 
        if use_expert_choice:
            self.moe = EC_SparseMoeBlock(hidden_size, mlp_dim, num_experts, float(num_experts_per_tok), pretraining_tp, num_shared_experts=num_shared_experts)
        else:
            self.moe = TC_SparseMoeBlock(hidden_size, mlp_dim, num_experts, int(num_experts_per_tok), pretraining_tp, num_shared_experts=num_shared_experts)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)) 
        x = x + gate_mlp.unsqueeze(1) * self.moe(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe, dropout=(self.dropout if self.training else 0.0))
        x = self.proj(x)
        return x

class DoubleStreamBlock(nn.Module):
    """
    A DiT block with seperate MoE for text & image
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int,
        num_experts=8,
        capacity_factor=2.0,
        pretraining_tp=2,
        num_shared_experts=2,
        dropout: float = 0.1,
        exp_ratio: int = 4,
        use_expert_choice: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, dropout=dropout)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, dropout=dropout)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        text_exps = max(1, num_experts // exp_ratio)
        if use_expert_choice:
            text_capacity = min(float(text_exps), float(capacity_factor))

            self.img_moe = EC_SparseMoeBlock(
                hidden_size, mlp_dim, num_experts, float(capacity_factor), pretraining_tp, num_shared_experts
            )
            self.txt_moe = EC_SparseMoeBlock(
                hidden_size, mlp_dim, text_exps, text_capacity, pretraining_tp, num_shared_experts
            )
        else:
            text_capacity = min(int(text_exps), int(capacity_factor))
            self.img_moe = TC_SparseMoeBlock(
                hidden_size, mlp_dim, num_experts, int(capacity_factor), pretraining_tp, num_shared_experts
            )
            self.txt_moe = TC_SparseMoeBlock(
                hidden_size, mlp_dim, text_exps, text_capacity, pretraining_tp, num_shared_experts
            )

    def forward(
        self,
        img: Tensor,          # [B, L_img, hidden_size]
        txt: Tensor,          # [B, L_txt, hidden_size]
        vec: Tensor,          # conditioning vector => Modulation
        pe: Tensor = None,    # rope positional encoding
    ) -> tuple[Tensor, Tensor]:
        dtype = img.dtype
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)


        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_moe(((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift).to(dtype))

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_moe(((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift).to(dtype))
        
        return img, txt
    
class SingleStreamBlock(nn.Module):
    """
    A DiT "single-stream" block with:
      - separate text/image QKV
      - a single cross-attention pass over concatenated sequences
      - Sparse MoE in place of the original MLP
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = mlp_dim
        self.dropout = dropout

        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(
        self,
        x: Tensor,   # [B, L_img + L_txt, hidden_size]
        vec: Tensor,   # conditioning vector => for Modulation
        pe: Tensor = None,    # rope positional encoding
    ) -> tuple[Tensor, Tensor]:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        qkv = qkv.contiguous()
        mlp = mlp.contiguous()

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, dropout=(self.dropout if self.training else 0.0))
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output
