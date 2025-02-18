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

class EC_MoEGate(nn.Module):
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
            topk_idx:    LongTensor of shape (B, E, C)
            topk_weight: FloatTensor of shape (B, E, C)
        """
        B, S, d = hidden_states.shape

        # Compute logits for each token: shape (B, S, E)
        logits = F.linear(hidden_states, self.weight)  # no bias

        # Apply softmax over experts for each token: shape (B, S, E)
        scores = logits.softmax(dim=-1)

        # Transpose to shape (B, E, S) so that for each batch and each expert we have S scores
        scores = scores.transpose(1, 2)  # now (B, E, S)

        # Compute per-example capacity: each expert picks top C tokens per example
        capacity = int(S * self.f_c / self.num_experts)

        # Each expert in each batch element picks top-C tokens over the S tokens
        topk_weight, topk_idx = torch.topk(scores, k=capacity, dim=-1, sorted=False)
        # topk_weight and topk_idx: (B, E, C)

        return topk_idx, topk_weight


class EC_SparseMoeBlock(nn.Module):
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
        self.gate = EC_MoEGate(
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

        # Gate returns indices of shape (B, E, C)
        topk_idx, topk_weight = self.gate(hidden_states)

        # Create an output buffer for the flattened tokens: (B*S, d)
        flat_out = torch.zeros(B * S, d, device=hidden_states.device, dtype=hidden_states.dtype)

        # Compute an offset for each batch element so that indices point to the correct positions
        batch_offset = (torch.arange(B, device=hidden_states.device) * S).view(B, 1, 1)
        # Adjust the token indices to be in the flattened space: shape (B, E, C)
        flat_topk_idx = topk_idx + batch_offset

        # Flatten hidden_states: (B*S, d)
        flat_x = hidden_states.view(B * S, d)

        # Process each expert (the gate still has shape (B, E, C))
        for expert_idx, expert_layer in enumerate(self.experts):
            # Extract indices and weights for expert_idx across the entire batch: shapes (B, C)
            expert_indices = flat_topk_idx[:, expert_idx, :]  # (B, C)
            expert_weights = topk_weight[:, expert_idx, :]      # (B, C)

            # Flatten these so that we process all selected tokens at once: (B*C,)
            expert_indices_flat = expert_indices.reshape(-1)
            expert_weights_flat = expert_weights.reshape(-1, 1)  # (B*C, 1)

            # Gather tokens from the flattened hidden states: (B*C, d)
            expert_tokens = flat_x[expert_indices_flat]
            
            # Process the tokens through the expert
            expert_out = expert_layer(expert_tokens)
            
            # Weight the outputs by the gating weights
            expert_out = expert_out * expert_weights_flat

            # Scatter-add back into the flat output buffer
            flat_out.index_add_(0, expert_indices_flat, expert_out)

        # Reshape the output back to (B, S, d)
        out = flat_out.view(B, S, d)

        # Optionally add the shared-expert MLP
        if self.n_shared_experts is not None:
            out = out + self.shared_experts(identity)

        return out

### Token choice
class TC_MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape    
        # print(bsz, seq_len, h)    
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class TC_SparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, embed_dim, mlp_ratio=4, num_experts=16, num_experts_per_tok=2, pretraining_tp=2, num_shared_experts=2):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList([MoeMLP(hidden_size = embed_dim, intermediate_size = mlp_ratio * embed_dim, pretraining_tp=pretraining_tp) for i in range(num_experts)])
        self.gate = TC_MoEGate(embed_dim=embed_dim, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.n_shared_experts = num_shared_experts
        
        if self.n_shared_experts is not None:
            intermediate_size =  embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(hidden_size = embed_dim, intermediate_size = intermediate_size, pretraining_tp=pretraining_tp)
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states) 
        # print(topk_idx.tolist(), print(len(topk_idx.tolist()))) 
        # global selected_ids_list
        # selected_ids_list.append(topk_idx.tolist())

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts): 
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y
    

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x) 
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok 
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]]) 
            
            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache

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
from timm.models.vision_transformer import Attention

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self, hidden_size, num_heads, mlp_ratio=4,
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
            self.moe = EC_SparseMoeBlock(hidden_size, mlp_ratio, num_experts, float(num_experts_per_tok), pretraining_tp, num_shared_experts=num_shared_experts)
        else:
            self.moe = TC_SparseMoeBlock(hidden_size, mlp_ratio, num_experts, int(num_experts_per_tok), pretraining_tp, num_shared_experts=num_shared_experts)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)) 
        x = x + gate_mlp.unsqueeze(1) * self.moe(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DoubleStreamBlock(nn.Module):
    """
    A DiT block with MoE for text & image, plus separate 1D & 2D RoPE application
    and joint attention. Similar to flux implementation, but with MoE instead of MLP.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int = 4,
        num_experts=8,
        capacity_factor=2.0,
        pretraining_tp=2,
        num_shared_experts=2,
        dropout: float = 0.1,
        exp_ratio: int = 4,
        use_expert_choice: bool = False,
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
        if use_expert_choice:
            text_capacity = min(float(text_exps), float(capacity_factor))

            self.img_moe = EC_SparseMoeBlock(
                hidden_size, mlp_ratio, num_experts, float(capacity_factor), pretraining_tp, num_shared_experts
            )
            self.txt_moe = EC_SparseMoeBlock(
                hidden_size, mlp_ratio, text_exps, text_capacity, pretraining_tp, num_shared_experts
            )
        else:
            text_capacity = min(int(text_exps), int(capacity_factor))
            self.img_moe = TC_SparseMoeBlock(
                hidden_size, mlp_ratio, num_experts, int(capacity_factor), pretraining_tp, num_shared_experts
            )
            self.txt_moe = TC_SparseMoeBlock(
                hidden_size, mlp_ratio, text_exps, text_capacity, pretraining_tp, num_shared_experts
            )

    def forward(
        self,
        img: torch.Tensor,          # [B, L_img, hidden_size]
        txt: torch.Tensor,          # [B, L_txt, hidden_size]
        vec: torch.Tensor,          # conditioning vector => Modulation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns updated (img, txt).
        We do:
          1) image attn with MoE
          2) text attn with MoE
          3) joint attention of (txt+img)
        """
        B, L_txt, _ = txt.shape

        # 1) modulate image
        img_mod1, img_mod2 = self.img_mod(vec)  
        img_in = self.img_norm1(img)
        img_in = (1 + img_mod1.scale) * img_in + img_mod1.shift

        # 2) get q,k,v for image
        img_qkv = self.img_qkv(img_in)  # [B, L_img, 3*hidden_size]
        img_q, img_k, img_v = rearrange(img_qkv, "B L (three H D) -> three B H L D", three=3, H=self.num_heads)
        # QK-norm
        img_q, img_k = self.img_qk_norm(img_q, img_k, img_v)

        # 3) modulate text
        txt_mod1, txt_mod2 = self.txt_mod(vec)  
        txt_in = self.txt_norm1(txt)
        txt_in = (1 + txt_mod1.scale) * txt_in + txt_mod1.shift

        # 4) get q,k,v for text
        txt_qkv = self.txt_qkv(txt_in) 
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (three H D) -> three B H L D", three=3, H=self.num_heads)
        txt_q, txt_k = self.txt_qk_norm(txt_q, txt_k, txt_v)

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
        mlp_ratio: int = 4,
        num_experts: int = 8,
        capacity_factor: int = 2,
        pretraining_tp: int = 2,
        num_shared_experts: int = 2,
        dropout: float = 0.1,
        use_expert_choice: bool = False,
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

        if use_expert_choice:
            self.moe = EC_SparseMoeBlock(
                hidden_size,
                mlp_ratio,
                num_experts,
                float(capacity_factor),
                pretraining_tp,
                num_shared_experts,
            )
        else:
            self.moe = TC_SparseMoeBlock(
                hidden_size,
                mlp_ratio,
                num_experts,
                int(capacity_factor),
                pretraining_tp,
                num_shared_experts,
            )

    def forward(
        self,
        img: torch.Tensor,   # [B, L_img, hidden_size]
        txt: torch.Tensor,   # [B, L_txt, hidden_size]
        vec: torch.Tensor,   # conditioning vector => for Modulation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L_txt, _ = txt.shape
        _, L_img, _ = img.shape

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

        # ---- IMAGE branch ----
        img_in = self.norm(img)
        img_in = (1 + mod1.scale) * img_in + mod1.shift
        img_qkv = self.qkv(img_in)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (three H D) -> three B H L D", three=3, H=self.num_heads
        )
        img_q, img_k = self.qk_norm(img_q, img_k, img_v)

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
