# Original code: https://github.com/feizc/DiT-MoE/blob/main/models.py

from dataclasses import dataclass
from einops import rearrange
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from transformer.utils import remove_masked_tokens
from transformer.embed import rope_1d, rope_2d

#################################################################################
#                                MoE Layer.                                     #
#################################################################################
class MoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, pretraining_tp=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()
        self.pretraining_tp = pretraining_tp

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0) 
            # print(self.up_proj.weight.size(), self.down_proj.weight.size())
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=-1)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class EC_DiTMoEGate(nn.Module):
    """
    Expert-Choice gating without auxiliary loss.
    Each expert will pick the top-C tokens, where
      C = (S * f_c) / E,
    computed in the forward pass.
    """
    def __init__(self, embed_dim, num_experts=16, f_c=2.0):
        super().__init__()
        self.num_experts = num_experts
        self.f_c = f_c  # capacity factor

        # gating weight: shape (E, d)
        self.weight = nn.Parameter(torch.empty(num_experts, embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        hidden_states shape: (B, S, d)
        Returns:
            topk_idx:    LongTensor of shape (E, C)
            topk_weight: FloatTensor of shape (E, C)
        """
        B, S, d = hidden_states.shape

        # 1) Flatten to (B*S, d)
        flat_x = hidden_states.reshape(-1, d)  # shape: (B*S, d)

        # 2) Compute token-expert logits => shape: (B*S, E)
        logits = F.linear(flat_x, self.weight)  # no bias

        # 3) Softmax across experts => shape: (B*S, E)
        scores = logits.softmax(dim=-1)

        # 4) Transpose to (E, B*S) so each expert can pick top-C tokens
        scores_t = scores.transpose(0, 1)  # => (E, B*S)

        # 5) Compute capacity = floor or ceil as you prefer
        #    Using integer floor here:
        capacity = int(B * S * self.f_c / self.num_experts)
        # (If you want to account for batch size, do: capacity = int(B*S*self.f_c / self.num_experts))

        # 6) Each expert picks top-C tokens
        topk_weight, topk_idx = torch.topk(scores_t, k=capacity, dim=-1, sorted=False)
        # shapes: both (E, C)

        return topk_idx, topk_weight

class SparseMoeBlock(nn.Module):
    """
    Expert-Choice sparse MoE block with no auxiliary load-balancing loss.
    """
    def __init__(
        self,
        embed_dim,
        mlp_ratio=4,
        num_experts=16,
        f_c=2.0,
        pretraining_tp=2,
        num_shared_experts=2
    ):
        super().__init__()
        self.num_experts = num_experts
        self.f_c = f_c

        # Create experts
        self.experts = nn.ModuleList([
            MoeMLP(
                hidden_size=embed_dim,
                intermediate_size=mlp_ratio * embed_dim,
                pretraining_tp=pretraining_tp
            )
            for _ in range(num_experts)
        ])

        # Our updated gate with no aux loss
        self.gate = EC_DiTMoEGate(
            embed_dim=embed_dim,
            num_experts=num_experts,
            f_c=f_c
        )

        # Shared "expert"
        self.n_shared_experts = num_shared_experts
        if self.n_shared_experts is not None:
            intermediate_size = embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(
                hidden_size=embed_dim,
                intermediate_size=intermediate_size,
                pretraining_tp=pretraining_tp
            )

    def forward(self, hidden_states):
        """
        hidden_states: (B, S, d)
        """
        B, S, d = hidden_states.shape
        identity = hidden_states

        # 1) Gate: each expert selects top-C tokens
        topk_idx, topk_weight = self.gate(hidden_states)

        # 2) Flatten tokens
        flat_x = hidden_states.view(-1, d)  # shape: (B*S, d)

        # 3) Initialize output buffer
        flat_out = torch.zeros_like(flat_x)

        # 4) For each expert, gather, process, scatter
        for expert_idx, expert_layer in enumerate(self.experts):
            # token indices this expert claims
            token_indices = topk_idx[expert_idx]   # shape: (C,)

            # gather tokens
            expert_tokens = flat_x[token_indices]  # shape: (C, d)

            # forward through the expert
            expert_out = expert_layer(expert_tokens)

            # multiply by gating weight
            w = topk_weight[expert_idx].unsqueeze(-1)  # shape: (C, 1)
            expert_out = expert_out * w

            # scatter-add back
            flat_out.index_add_(0, token_indices, expert_out)

        # 5) Reshape to original shape
        out = flat_out.view(B, S, d)

        # 6) Optional: add shared-expert MLP on top
        if self.n_shared_experts is not None:
            out = out + self.shared_experts(identity)

        return out



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: torch.Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.gelu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )
    

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout=0.0) -> torch.Tensor:
    """
    Scaled dot-product attention with shape conventions:
      q, k, v: [B, H, L, D]
        B = batch size
        H = number of heads
        L = sequence length
        D = head dimension
    Returns:
      x: [B, L, H*D], sequence-first
    """
    # Use PyTorch 2.0 built-in scaled_dot_product_attention
    # which expects q,k,v: [B, H, L, D]
    # and returns [B, H, L, D]
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)  # shape [B, H, L, D]
    
    # Rearrange to [B, L, H*D]
    x = rearrange(x, "B H L D -> B L (H D)")
    return x

#################################################################################
#                                 Core DiT Modules                              #
#################################################################################

class DoubleStreamBlock(nn.Module):
    """
    A DiT block with MoE for text & image, plus separate 1D & 2D RoPE application
    and joint attention. Similar to flux implementation, but with MoE instead of MLP.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_experts=8,
        num_experts_per_tok=2,
        pretraining_tp=2,
        num_shared_experts=2,
        dropout: float = 0.1,
        exp_ratio: int = 4
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        # Norm + SelfAttention for image
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.img_qk_norm = QKNorm(self.head_dim)
        self.img_proj = nn.Linear(hidden_size, hidden_size)

        # Norm + SelfAttention for text
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.txt_qk_norm = QKNorm(self.head_dim)
        self.txt_proj = nn.Linear(hidden_size, hidden_size)

        # MoE blocks instead of standard MLP
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        text_exps = max(1, num_experts // exp_ratio)

        self.img_moe = SparseMoeBlock(
            hidden_size, mlp_ratio, num_experts, num_experts_per_tok, pretraining_tp, num_shared_experts
        )
        self.txt_moe = SparseMoeBlock(
            hidden_size, mlp_ratio, text_exps, num_experts_per_tok, pretraining_tp, num_shared_experts
        )

    def forward(
        self,
        img: torch.Tensor,          # [B, L_img, hidden_size]
        txt: torch.Tensor,          # [B, L_txt, hidden_size]
        vec: torch.Tensor,          # conditioning vector => Modulation
        mask: torch.Tensor,
        h: int,
        w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns updated (img, txt).
        We do:
          1) image attn with MoE
          2) text attn with MoE
          3) joint attention of (txt+img)
        """
        B, L_txt, _ = txt.shape

        # (seq_len, pos_emb_dim)
        txt_rope = rope_1d(self.head_dim, L_txt)
        # (batch_size, seq_len, pos_emb_dim)
        txt_rope = txt_rope.unsqueeze(0).repeat(B, 1, 1).to(img.device)

        # (height, width, embed_dim)
        img_rope = rope_2d(self.head_dim, h, w)
        # (batch_size, height*width, pos_emb_dim)
        img_rope = img_rope.unsqueeze(0).repeat(B, 1, 1).to(img.device)
        if mask is not None:
            img_rope = remove_masked_tokens(img_rope, mask)

        # 1) modulate image
        img_mod1, img_mod2 = self.img_mod(vec)  
        img_in = self.img_norm1(img)
        img_in = (1 + img_mod1.scale) * img_in + img_mod1.shift

        # 2) get q,k,v for image
        img_qkv = self.img_qkv(img_in)  # [B, L_img, 3*hidden_size]
        img_q, img_k, img_v = rearrange(img_qkv, "B L (three H D) -> three B H L D", three=3, H=self.num_heads)
        # QK-norm
        img_q, img_k = self.img_qk_norm(img_q, img_k, img_v)
        # apply 2D rope => shape [B, L_img, hidden_size]
        img_q, img_k = rotary_multiply_2d(img_q, img_k, img_rope)

        # 3) modulate text
        txt_mod1, txt_mod2 = self.txt_mod(vec)  
        txt_in = self.txt_norm1(txt)
        txt_in = (1 + txt_mod1.scale) * txt_in + txt_mod1.shift

        # 4) get q,k,v for text
        txt_qkv = self.txt_qkv(txt_in) 
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (three H D) -> three B H L D", three=3, H=self.num_heads)
        txt_q, txt_k = self.txt_qk_norm(txt_q, txt_k, txt_v)
        # apply 1D rope
        txt_q, txt_k = rotary_multiply_1d(txt_q, txt_k, txt_rope)

        # 5) joint attention
        # Cat along the L dimension
        q = torch.cat([txt_q, img_q], dim=2)  # [B,H, L_txt+L_img, D]
        k = torch.cat([txt_k, img_k], dim=2)
        v = torch.cat([txt_v, img_v], dim=2)

        # Standard scaled dot-product attention:
        attn_out = attention(q, k, v, self.dropout)  # shape [B, L_txt+L_img, D]
        txt_attn = attn_out[:, : txt.shape[1]]  # first L_txt
        img_attn = attn_out[:, txt.shape[1] :]

        # 6) final projections
        #   - For image
        img_out = img + img_mod1.gate * self.img_proj(img_attn)
        # MoE for image
        img_out = img_out + img_mod2.gate * self.img_moe(
            (1 + img_mod2.scale) * self.img_norm2(img_out) + img_mod2.shift
        )

        #   - For text
        txt_out = txt + txt_mod1.gate * self.txt_proj(txt_attn)
        # MoE for text
        txt_out = txt_out + txt_mod2.gate * self.txt_moe(
            (1 + txt_mod2.scale) * self.txt_norm2(txt_out) + txt_mod2.shift
        )

        return img_out, txt_out
    
class SingleStreamBlock(nn.Module):
    """
    A DiT "single-stream" block with:
      - separate text/image QKV + RoPE
      - a single cross-attention pass over concatenated sequences
      - Sparse MoE in place of the original MLP
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        pretraining_tp: int = 2,
        num_shared_experts: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        # Modulation with "double=True" so we get two sets (mod1, mod2)
        #   - typically one set is used for the attn skip-connection
        #   - the second set is used for the MoE skip-connection
        self.modulation = Modulation(hidden_size, double=True)

        # First norm + linear for text
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.qk_norm = QKNorm(self.head_dim)

        # Output projection after attention
        self.proj = nn.Linear(hidden_size, hidden_size)

        # Second norm and sparse MoE, replacing what was originally an MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.moe = SparseMoeBlock(
            hidden_size,
            mlp_ratio,
            num_experts,
            num_experts_per_tok,
            pretraining_tp,
            num_shared_experts,
        )

    def forward(
        self,
        img: torch.Tensor,   # [B, L_img, hidden_size]
        txt: torch.Tensor,   # [B, L_txt, hidden_size]
        vec: torch.Tensor,   # conditioning vector => for Modulation
        mask: torch.Tensor,  # optional mask for images (or None)
        h: int,              # image height (2D RoPE)
        w: int,              # image width  (2D RoPE)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L_txt, _ = txt.shape
        _, L_img, _ = img.shape

        # 1) Prepare text RoPE (1D)
        rope_txt = rope_1d(self.head_dim, L_txt)  # [L_txt, d]
        rope_txt = rope_txt.unsqueeze(0).repeat(B, 1, 1).to(txt.device)

        # 2) Prepare image RoPE (2D)
        rope_img = rope_2d(self.head_dim, h, w)   # [L_img, d], flattened from (h*w)
        rope_img = rope_img.unsqueeze(0).repeat(B, 1, 1).to(img.device)
        if mask is not None:
            rope_img = remove_masked_tokens(rope_img, mask)

        # 3) modulation parameters => "double=True" => mod1, mod2
        mod1, mod2 = self.modulation(vec)

        # ---- TEXT branch ----
        txt_in = self.norm(txt)
        txt_in = (1 + mod1.scale) * txt_in + mod1.shift
        txt_qkv = self.qkv(txt_in)  # [B, L_txt, 3*hidden_size]
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (three H D) -> three B H L D", three=3, H=self.num_heads
        )
        txt_q, txt_k = self.qk_norm(txt_q, txt_k, txt_v)
        # Apply 1D RoPE
        txt_q, txt_k = rotary_multiply_1d(txt_q, txt_k, rope_txt)

        # ---- IMAGE branch ----
        img_in = self.norm(img)
        img_in = (1 + mod1.scale) * img_in + mod1.shift
        img_qkv = self.qkv(img_in)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (three H D) -> three B H L D", three=3, H=self.num_heads
        )
        img_q, img_k = self.qk_norm(img_q, img_k, img_v)
        # Apply 2D RoPE
        img_q, img_k = rotary_multiply_2d(img_q, img_k, rope_img)

        # ---- Single-stream attention: concat txt + img ----
        q = torch.cat([txt_q, img_q], dim=2)  # [B, H, (L_txt+L_img), D]
        k = torch.cat([txt_k, img_k], dim=2)
        v = torch.cat([txt_v, img_v], dim=2)

        # Standard scaled dot-product attention across entire sequence
        attn_out = attention(q, k, v, dropout=self.dropout)
        # Split out txt/img
        txt_attn = attn_out[:, :L_txt]  # [B, L_txt, D]
        img_attn = attn_out[:, L_txt:]  # [B, L_img, D]

        # ---- Add first skip-connection + projection ----
        txt_out = txt + mod1.gate * self.proj(txt_attn)
        img_out = img + mod1.gate * self.proj(img_attn)

        # ---- Single MoE over entire sequence ----
        #   (We re-concat txt + img, run LN+shift+scale, pass to MoE, then split)
        x_out = torch.cat([txt_out, img_out], dim=1)  # [B, L_txt+L_img, hidden_size]
        x_in = self.norm2(x_out)
        x_in = (1 + mod2.scale) * x_in + mod2.shift

        x_moe = self.moe(x_in)
        x_out = x_out + mod2.gate * x_moe

        # final separation
        txt_out = x_out[:, :L_txt]
        img_out = x_out[:, L_txt:]

        return img_out, txt_out

## Rope helper functions

def rotary_multiply_1d(
    q: torch.Tensor, 
    k: torch.Tensor, 
    rope_1d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies standard 1D rotary embedding to (q,k).
    - q, k: shape [B, H, L, D]
    - rope_1d: shape [B, L, D], i.e. cos/sin interleaved
    Returns:
        (q_rotated, k_rotated) of the same shape.
    """
    B, H, L, D = q.shape

    # Reshape rope to [B, L, D/2, 2] => split into cos and sin
    rope_2 = rope_1d.view(B, L, D // 2, 2).float()

    q_2 = q.view(B, H, L, D // 2, 2).float()
    k_2 = k.view(B, H, L, D // 2, 2).float()

    cos_ = rope_2[..., 0].unsqueeze(1)  # => [B, 1, L, D//2]
    sin_ = rope_2[..., 1].unsqueeze(1)

    # standard rotation: (x, y) -> (x cos - y sin, x sin + y cos)
    x_q, y_q = q_2[..., 0], q_2[..., 1]
    x_k, y_k = k_2[..., 0], k_2[..., 1]

    q_rot = torch.stack([
        x_q * cos_ - y_q * sin_,
        x_q * sin_ + y_q * cos_
    ], dim=-1)
    k_rot = torch.stack([
        x_k * cos_ - y_k * sin_,
        x_k * sin_ + y_k * cos_
    ], dim=-1)

    # reshape back
    q_out = q_rot.view(B, H, L, D).type_as(q)
    k_out = k_rot.view(B, H, L, D).type_as(k)
    return q_out, k_out


def rotary_multiply_2d(
    q: torch.Tensor, 
    k: torch.Tensor, 
    rope_2d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 2D rotary embedding by splitting row vs. col halves.
    - q, k: shape [B, H, L, D]
    - rope_2d: shape [B, L, D], where the first half is row-emb, second half is col-emb
    Returns:
        (q_rotated, k_rotated) of the same shape.
    """
    B, H, L, D_total = q.shape
    half_dim = D_total // 2

    # The first half of rope_2d => row part
    # The second half => col part
    row_part = rope_2d[:, :, :half_dim]
    col_part = rope_2d[:, :, half_dim:]

    # Split q, k along their last dimension into row vs. col portions
    q_row, q_col = q.split(half_dim, dim=-1)
    k_row, k_col = k.split(half_dim, dim=-1)

    # Now rotate row half with row_part, col half with col_part
    q_row_out, k_row_out = rotary_multiply_1d(q_row, k_row, row_part)
    q_col_out, k_col_out = rotary_multiply_1d(q_col, k_col, col_part)

    # Concatenate the row-rotated and col-rotated
    q_out = torch.cat([q_row_out, q_col_out], dim=-1)
    k_out = torch.cat([k_row_out, k_col_out], dim=-1)
    return q_out, k_out