from transformer.moedit import DoubleStreamBlock, SingleStreamBlock
from torch import nn
from dataclasses import dataclass

@dataclass
class BackboneParams:
    input_dim: int
    embed_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
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
        # Define scaling ranges for m_f and m_a
        ma_min, ma_max = 0.5, 1.0

        self.layers = nn.ModuleList()
        for i in range(params.num_layers):
            # Calculate scaling factors for the i-th layer using linear interpolation
            ma = ma_min + (ma_max - ma_min) * i / (params.num_layers - 1)

            scaled_num_heads = max(1, int(params.num_heads * ma))
            scaled_num_heads = nearest_divisor(scaled_num_heads, params.embed_dim)

            mlp_ratio = max(1, int(params.mlp_dim / params.embed_dim))

            if i % 2 == 0:  # Even layers use regular DiT (no MoE)
                n_exp = 1
                n_shared = 1
                n_act = 1.0
            else:  # Odd layers use MoE DiT
                n_exp = params.num_experts
                n_shared = params.shared_experts
                n_act = params.capacity_factor

            if i < params.num_layers // 2: # First half uses DoubleStreamBlock
                self.layers.append(DoubleStreamBlock(params.embed_dim, scaled_num_heads, mlp_ratio, 
                                            n_exp, n_act, pretraining_tp=params.pretraining_tp, num_shared_experts=n_shared, dropout=params.dropout, exp_ratio=params.image_text_expert_ratio))
            else:  # Second half uses SingleStreamBlock
                self.layers.append(SingleStreamBlock(params.embed_dim, scaled_num_heads, mlp_ratio, 
                                               n_exp, n_act, pretraining_tp=params.pretraining_tp, num_shared_experts=n_shared, dropout=params.dropout))


    def forward(self, x, text, vec, mask, original_h, original_w):
        for layer in self.layers:
            x, text = layer(x, text, vec, mask, original_h, original_w)

        return x