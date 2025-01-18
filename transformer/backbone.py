from transformer.moedit import DoubleStreamBlock, SingleStreamBlock
from torch import nn

def nearest_divisor(scaled_num_heads, embed_dim):
    # Find all divisors of embed_dim
    divisors = [i for i in range(1, embed_dim + 1) if embed_dim % i == 0]
    
    # Find the nearest divisor
    nearest = min(divisors, key=lambda x: abs(x - scaled_num_heads))
    
    return nearest

class TransformerBackbone(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, cond_embed: int, num_layers: int, num_heads: int, mlp_dim: int, 
                 num_experts: int = 4, active_experts: int = 2, shared_experts: int = 2, dropout: float = 0.1, pretraining_tp: int = 1):
        super().__init__()
        self.image_embedding = nn.Linear(input_dim, embed_dim)
        self.text_embedding = nn.Linear(cond_embed, embed_dim)

        # Define scaling ranges for m_f and m_a
        mf_min, mf_max = 0.5, 4.0
        ma_min, ma_max = 0.5, 1.0

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Calculate scaling factors for the i-th layer using linear interpolation
            mf = mf_min + (mf_max - mf_min) * i / (num_layers - 1)
            ma = ma_min + (ma_max - ma_min) * i / (num_layers - 1)

            # Scale the dimensions according to the scaling factors
            scaled_mlp_dim = int(mlp_dim * mf)
            scaled_num_heads = max(1, int(num_heads * ma))
            scaled_num_heads = nearest_divisor(scaled_num_heads, embed_dim)
            mlp_ratio = int(scaled_mlp_dim / embed_dim)

            if i % 2 == 0:  # Even layers use regular DiT (no MoE)
                n_exp = 1
                n_shared = 1
                n_act = 1
            else:  # Odd layers use MoE DiT
                n_exp = num_experts
                n_shared = shared_experts
                n_act = active_experts

            if i < num_layers // 2: # First half uses DoubleStreamBlock
                self.layers.append(DoubleStreamBlock(embed_dim, scaled_num_heads, mlp_ratio, 
                                            n_exp, n_act, pretraining_tp=pretraining_tp, num_shared_experts=n_shared, dropout=dropout))
            else:  # Second half uses SingleStreamBlock
                self.layers.append(SingleStreamBlock(embed_dim, scaled_num_heads, mlp_ratio, 
                                               n_exp, n_act, pretraining_tp=pretraining_tp, num_shared_experts=n_shared, dropout=dropout))

        self.output_layer = nn.Linear(embed_dim, input_dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Apply basic initialization to all modules
        self.apply(_basic_init)

        # Initialize DiTBlocks if any
        for layer in self.layers:
            layer.initialize_weights()

    def forward(self, x, text, vec, mask, original_h, original_w):
        x = self.image_embedding(x)
        text_emb = self.text_embedding(text)

        for layer in self.layers:
            x, text_emb = layer(x, text_emb, vec, mask, original_h, original_w)

        x = self.output_layer(x)
        return x