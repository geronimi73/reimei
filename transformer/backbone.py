from transformer.moedit import DiTBlock
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

            # Choose layer type based on the layer index (even/odd)
            if i % 2 == 0:  # Even layers use regular DiT
                self.layers.append(DiTBlock(embed_dim, scaled_num_heads, mlp_ratio, 
                                            1, 1, pretraining_tp=pretraining_tp, num_shared_experts=shared_experts, dropout=dropout))
            else:  # Odd layers use MoE DiT
                self.layers.append(DiTBlock(embed_dim, scaled_num_heads, mlp_ratio, 
                                               num_experts, active_experts, pretraining_tp=pretraining_tp, num_shared_experts=shared_experts, dropout=dropout))

        self.output_layer = nn.Linear(embed_dim, input_dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.TransformerEncoderLayer):
                # Initialize TransformerEncoderLayer modules
                # Initialize the self-attention layers
                nn.init.xavier_uniform_(module.self_attn.in_proj_weight)
                if module.self_attn.in_proj_bias is not None:
                    nn.init.constant_(module.self_attn.in_proj_bias, 0)
                nn.init.xavier_uniform_(module.self_attn.out_proj.weight)
                if module.self_attn.out_proj.bias is not None:
                    nn.init.constant_(module.self_attn.out_proj.bias, 0)
                # Initialize the linear layers in the feedforward network
                for lin in [module.linear1, module.linear2]:
                    nn.init.xavier_uniform_(lin.weight)
                    if lin.bias is not None:
                        nn.init.constant_(lin.bias, 0)
                # Initialize the LayerNorm layers
                for ln in [module.norm1, module.norm2]:
                    if ln.weight is not None:
                        nn.init.constant_(ln.weight, 1.0)
                    if ln.bias is not None:
                        nn.init.constant_(ln.bias, 0)

        # Apply basic initialization to all modules
        self.apply(_basic_init)

        # Initialize input and class embeddings
        nn.init.xavier_uniform_(self.image_embedding.weight)
        if self.image_embedding.bias is not None:
            nn.init.constant_(self.image_embedding.bias, 0)
        nn.init.xavier_uniform_(self.text_embedding.weight)
        if self.text_embedding.bias is not None:
            nn.init.constant_(self.text_embedding.bias, 0)

        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

        # Initialize DiTBlocks if any
        for layer in self.layers:
            if isinstance(layer, DiTBlock):
                layer.initialize_weights()

    def forward(self, x, text, image_pos_emb, text_pos_emb):
        x = self.image_embedding(x)
        text_emb = self.text_embedding(text)

        for layer in self.layers:
            x, text = layer(x, text, image_pos_emb, text_pos_emb)

        x = self.output_layer(x)
        return x