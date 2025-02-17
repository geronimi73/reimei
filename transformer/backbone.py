import torch
from transformer.moedit import DoubleStreamBlock, SingleStreamBlock, DiTBlock
from torch import nn
from dataclasses import dataclass

@dataclass
class BackboneParams:
    input_dim: int
    embed_dim: int
    num_layers: int
    num_heads: int
    num_experts: int = 4
    capacity_factor: float = 2.0
    shared_experts: int = 2
    dropout: float = 0.1
    image_text_expert_ratio: int = 4
    pretraining_tp: int = 1

def nearest_divisor(scaled_num_heads, embed_dim):
    # Find all divisors of embed_dim
    divisors = [i for i in range(1, embed_dim + 1) if embed_dim % i == 0]
    
    # Find the nearest divisor
    nearest = min(divisors, key=lambda x: abs(x - scaled_num_heads))
    
    return nearest

class TransformerBackbone(nn.Module):
    def __init__(self, params: BackboneParams):
        super().__init__()
        mf_min, mf_max = 1.0, 4.0
        
        self.layers = nn.ModuleList()
        for i in range(params.num_layers):
            # Calculate scaling factors for the i-th layer using linear interpolation
            mf = mf_min + (mf_max - mf_min) * i / (params.num_layers - 1)

            # Scale the dimensions according to the scaling factors
            scaled_mlp_dim = (int(params.embed_dim * mf) // params.num_heads) * params.num_heads
            scaled_num_heads = params.num_heads
            mlp_ratio = max(1, int(scaled_mlp_dim / params.embed_dim))

            if i % 2 == 0:  # Even layers use regular DiT (no MoE)
                n_exp = 1
                n_shared = 1
                n_act = 1.0
            else:  # Odd layers use MoE DiT
                n_exp = params.num_experts
                n_shared = params.shared_experts
                n_act = min(params.capacity_factor, float(n_exp))
            
            # self.layers.append(DiTBlock(params.embed_dim, scaled_num_heads, mlp_ratio, 
                                # n_exp, n_act, pretraining_tp=params.pretraining_tp, num_shared_experts=n_shared, attn_drop=params.dropout))

            if i < params.num_layers // 2: # First half uses DoubleStreamBlock
                self.layers.append(DoubleStreamBlock(params.embed_dim, scaled_num_heads, mlp_ratio, 
                                            n_exp, n_act, pretraining_tp=params.pretraining_tp, num_shared_experts=n_shared, dropout=params.dropout, exp_ratio=params.image_text_expert_ratio))
            else:  # Second half uses SingleStreamBlock
                self.layers.append(SingleStreamBlock(params.embed_dim, scaled_num_heads, mlp_ratio, 
                                               n_exp, n_act, pretraining_tp=params.pretraining_tp, num_shared_experts=n_shared, dropout=params.dropout))


    def forward(
            self, 
            x: torch.Tensor,
            text: torch.Tensor,
            vec: torch.Tensor,
            img_rope: torch.Tensor,
            txt_rope: torch.Tensor,
            ):
        for layer in self.layers:
            x, text = layer(x, text, vec, img_rope, txt_rope)
            # x = layer(x, vec)

        return x