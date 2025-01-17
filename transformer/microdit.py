import torch.nn as nn
from .embed import rope_1d, rope_2d, TimestepEmbedder
from .utils import remove_masked_tokens, add_masked_tokens, unpatchify
from .backbone import TransformerBackbone
from .moedit import DiTBlock
import torch
from config import VAE_SCALING_FACTOR
from datasets import load_dataset

class TokenMixer(nn.Module):
    """
    Each layer expects:
        - img:       [B, L_img, embed_dim]
        - txt:       [B, L_txt, embed_dim]
        - vec:       [B, embed_dim]            (conditioning vector for Modulation)
        - img_rope:  [B, L_img, embed_dim]     (2D RoPE for images)
        - txt_rope:  [B, L_txt, embed_dim]     (1D RoPE for text)
    and returns the updated (img, txt) after `num_layers` of DiTBlock.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int = 2,
        mlp_ratio: int = 4,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        pretraining_tp: int = 2,
        num_shared_experts: int = 2
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DiTBlock(
                hidden_size=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                pretraining_tp=pretraining_tp,
                num_shared_experts=num_shared_experts
            )
            for _ in range(num_layers)
        ])

    def initialize_weights(self):
        """
        Initialize weights for each DiTBlock.
        """
        for layer in self.layers:
            layer.initialize_weights()

    def forward(
        self,
        img: torch.Tensor,       # [B, L_img, embed_dim]
        txt: torch.Tensor,       # [B, L_txt, embed_dim]
        vec: torch.Tensor,       # [B, embed_dim]
        img_rope: torch.Tensor,  # [B, L_img, pos_embed_dim]
        txt_rope: torch.Tensor   # [B, L_txt, pos_embed_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Passes (img, txt) through each DiTBlock in sequence. Each DiTBlock performs
        cross-attention over the concatenation of (img, txt), plus MoE-based feedforwards.
        """
        for layer in self.layers:
            img, txt = layer(img, txt, vec, img_rope, txt_rope)
        return img, txt

class MicroDiT(nn.Module):
    """
    MicroDiT is a image diffusion transformer model.

    Args:
        in_channels (int): Number of input channels in the image data.
        embed_dim (int): Dimension of the embedding space.
        num_layers (int): Number of layers in the transformer backbone.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        mlp_dim (int): Dimension of the multi-layer perceptron.
        num_experts (int, optional): Number of experts in the transformer backbone. Default is 4.
        active_experts (int, optional): Number of active experts in the transformer backbone. Default is 2.
        dropout (float, optional): Dropout rate. Default is 0.1.
        patch_mixer_layers (int, optional): Number of layers in the patch mixer. Default is 2.
        embed_cat (bool, optional): Whether to concatenate embeddings. Default is False. If true, the timestep, class, and positional embeddings are concatenated rather than summed.

    Attributes:
        embed_dim (int): Dimension of the embedding space.
        patch_embed (PatchEmbed): Patch embedding layer.
        time_embed (TimestepEmbedder): Timestep embedding layer.
        class_embed (nn.Sequential): Class embedding layer.
        mha (nn.MultiheadAttention): Multi-head attention mechanism.
        mlp (nn.Sequential): Multi-layer perceptron for processing embeddings.
        pool_mlp (nn.Sequential): Pooling and multi-layer perceptron for (MHA + MLP).
        linear (nn.Linear): Linear layer after MHA+MLP.
        patch_mixer (PatchMixer): Patch mixer layer.
        backbone (TransformerBackbone): Transformer backbone model.
    """
    def __init__(self, channels, embed_dim, num_layers, num_heads, mlp_dim, text_embed_dim,
                 num_experts=4, active_experts=2, shared_experts=2, dropout=0.1, patch_mixer_layers=2, embed_cat=False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pos_emb_dim = embed_dim // num_heads
        
        # Timestep embedding
        self.time_embed = TimestepEmbedder(self.embed_dim)
        
        # Caption embedding
        self.text_embed = nn.Sequential(
            nn.Linear(text_embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # MHA for timestep and caption
        self.mha = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)
        
        # MLP for timestep and caption
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Pool + MLP for (MHA + MLP)
        self.pool_mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Linear layer after MHA+MLP
        self.linear = nn.Linear(self.embed_dim, self.embed_dim)
        
        # TokenMixer
        self.token_mixer = TokenMixer(self.embed_dim, num_heads, patch_mixer_layers)
        
        # Backbone transformer model
        self.backbone = TransformerBackbone(self.embed_dim, self.embed_dim, self.embed_dim, num_layers, num_heads, mlp_dim, 
                                        num_experts, active_experts, shared_experts, dropout)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, channels)
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize all linear layers and biases
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                # Initialize convolutional layers like linear layers
                nn.init.xavier_uniform_(module.weight.view(module.weight.size(0), -1))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize MultiheadAttention layers
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                # Initialize LayerNorm layers
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

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        if self.time_embed.mlp[0].bias is not None:
            nn.init.constant_(self.time_embed.mlp[0].bias, 0)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        if self.time_embed.mlp[2].bias is not None:
            nn.init.constant_(self.time_embed.mlp[2].bias, 0)

        # Initialize caption embedding layers
        for layer in self.text_embed:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize MLP layers in self.mlp
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize MLP layers in self.pool_mlp
        for layer in self.pool_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize the linear layer in self.linear
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

        # Zero-out the last linear layer in the output to ensure initial predictions are zero
        nn.init.constant_(self.output[-1].weight, 0)
        if self.output[-1].bias is not None:
            nn.init.constant_(self.output[-1].bias, 0)

        self.backbone.initialize_weights()

        self.token_mixer.initialize_weights()

    def forward(self, x, t, text_embeddings, mask=None):
        # x: (batch_size, channels, height, width)
        # t: (batch_size, 1)
        # text_embeddings: (batch_size, text_embed_dim)
        # mask: (batch_size, num_tokens)
        
        batch_size, channels, height, width = x.shape

        # Reshape and transmute x to have shape (batch_size, height*width, channels)
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
        
        # Generate positional embeddings
        # (seq_len, pos_emb_dim)
        text_pos_emb = rope_1d(self.pos_emb_dim, text_embeddings.shape[1])
        # (batch_size, seq_len, pos_emb_dim)
        text_pos_emb = text_pos_emb.view(batch_size, text_embeddings.shape[1], self.pos_emb_dim)

        # (height, width, embed_dim)
        image_pos_emb = rope_2d(self.pos_emb_dim, height, width)
        # (batch_size, height*width, pos_emb_dim)
        image_pos_emb = image_pos_emb.view(batch_size, height * width, self.pos_emb_dim)
        
        # Timestep embedding
        t_emb = self.time_embed(t)  # (batch_size, embed_dim)

        # Caption embedding
        text_emb = self.text_embed(text_embeddings)  # (batch_size, embed_dim)

        mha_out = self.mha(t_emb.unsqueeze(1), text_emb.unsqueeze(1), text_emb.unsqueeze(1))[0].squeeze(1)
        mlp_out = self.mlp(mha_out)
        
        # Pool + MLP
        pool_out = self.pool_mlp(mlp_out.unsqueeze(2))

        # Pool + MLP + t_emb
        pool_out = (pool_out + t_emb).unsqueeze(1)
        
        # Apply linear layer
        cond_signal = self.linear(mlp_out).unsqueeze(1)  # (batch_size, 1, embed_dim)
        cond = (cond_signal + pool_out).expand(-1, x.shape[1], -1)
        
        # Add conditioning signal to all patches
        # (batch_size, num_tokens, embed_dim)
        x = x + cond

        # Patch-mixer
        x = self.patch_mixer(x, text_pos_emb, image_pos_emb)

        # Remove masked patches
        if mask is not None:
            x = remove_masked_tokens(x, mask)
            image_pos_emb = remove_masked_tokens(image_pos_emb, mask)

        # MHA + MLP + Pool + MLP + t_emb
        cond = (mlp_out.unsqueeze(1) + pool_out).expand(-1, x.shape[1], -1)

        x = x + cond

        # Backbone transformer model
        x = self.backbone(x, text_emb, text_pos_emb, image_pos_emb)
        
        # Final output layer
        # (bs, unmasked_num_tokens, embed_dim) -> (bs, unmasked_num_tokens, in_channels)
        x = self.output(x)

        # Add masked patches
        if mask is not None:
            # (bs, unmasked_num_tokens, in_channels) -> (bs, num_tokens, in_channels)
            x = add_masked_tokens(x, mask)

        x = x.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width).contiguous()
        
        return x
    
    @torch.no_grad()
    def sample(self, z, cond, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device).to(torch.float16)

            vc = self(z, t, cond, None)
            null_cond = torch.zeros_like(cond)
            vu = self(z, t, null_cond)
            vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return (images[-1] / VAE_SCALING_FACTOR)