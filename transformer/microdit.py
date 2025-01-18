import torch.nn as nn
from .embed import sincos_2d, TimestepEmbedder, MLPEmbedder, OutputLayer
from .utils import remove_masked_tokens, add_masked_tokens
from .backbone import TransformerBackbone
from .token_mixer import TokenMixer
import torch
from config import VAE_SCALING_FACTOR

class MicroDiT(nn.Module):
    """
    MicroDiT is a image diffusion transformer model.

    Args:
        channels (int): Number of input channels in the image data.
        embed_dim (int): Dimension of the embedding space.
        num_layers (int): Number of layers in the transformer backbone.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        mlp_dim (int): Dimension of the multi-layer perceptron.
        num_experts (int, optional): Number of experts in the transformer backbone. Default is 4.
        active_experts (int, optional): Number of active experts in the transformer backbone. Default is 2.
        shared_experts (int, optional): Number of shared experts in the transformer backbone. Default is 2.
        dropout (float, optional): Dropout rate. Default is 0.1.
        patch_mixer_layers (int, optional): Number of layers in the patch mixer. Default is 2.
        embed_cat (bool, optional): Whether to concatenate embeddings. Default is False. If true, the timestep, class, and positional embeddings are concatenated rather than summed.

    Attributes:
        embed_dim (int): Dimension of the embedding space.
        channels (int): Number of input channels in the image data.
        time_embedder (TimestepEmbedder): Timestep embedding layer.
        image_embedder (MLPEmbedder): Image embedding layer.
        text_embedder (MLPEmbedder): Text embedding layer.
        vector_embedder (MLPEmbedder): Vector embedding layer.
        token_mixer (TokenMixer): Token mixer layer.
        backbone (TransformerBackbone): Transformer backbone model.
        output (MLPEmbedder): Output layer.
    """
    def __init__(self, channels, embed_dim, num_layers, num_heads, mlp_dim, text_embed_dim, vector_embed_dim,
                 num_experts=4, active_experts=2, shared_experts=2, dropout=0.1, patch_mixer_layers=2, embed_cat=False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pos_emb_dim = embed_dim // num_heads
        self.channels = channels
        
        # Timestep embedding
        self.time_embedder = TimestepEmbedder(self.embed_dim)

        # Image embedding
        self.image_embedder = MLPEmbedder(self.channels, self.embed_dim)
        
        # Text embedding
        self.text_embedder = MLPEmbedder(text_embed_dim, self.embed_dim)

        # Vector (y) embedding
        self.vector_embedder = MLPEmbedder(vector_embed_dim, self.embed_dim)
        
        # TokenMixer
        self.token_mixer = TokenMixer(self.embed_dim, num_heads, patch_mixer_layers, num_experts=num_experts, num_experts_per_tok=active_experts)
        
        # Backbone transformer model
        self.backbone = TransformerBackbone(self.embed_dim, self.embed_dim, self.embed_dim, num_layers, num_heads, mlp_dim, 
                                        num_experts, active_experts, shared_experts, dropout)
        
        self.output_layer = OutputLayer(self.embed_dim, self.channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize all linear layers and biases
        def _basic_init(module):
            if isinstance(module, MLPEmbedder):
                # Initialize MLPEmbedder layers
                nn.init.xavier_uniform_(module.mlp[0].weight)
                nn.init.constant_(module.mlp[0].bias, 0)
                nn.init.xavier_uniform_(module.mlp[2].weight)
                nn.init.constant_(module.mlp[2].bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Apply basic initialization to all modules
        self.apply(_basic_init)

        # Timestep embedder
        nn.init.normal_(self.time_embedder.mlp.mlp[0].weight, std=0.02)
        nn.init.constant_(self.time_embedder.mlp.mlp[0].bias, 0)
        nn.init.normal_(self.time_embedder.mlp.mlp[2].weight, std=0.02)
        nn.init.constant_(self.time_embedder.mlp.mlp[2].bias, 0)

        # Zero-out the last linear layer in the output to ensure initial predictions are zero
        nn.init.constant_(self.output_layer.mlp.mlp[-1].weight, 0)
        nn.init.constant_(self.output_layer.mlp.mlp[-1].bias, 0)

        self.backbone.initialize_weights()
        self.token_mixer.initialize_weights()

    def forward(self, img, time, txt, vec, mask=None):
        # img: (batch_size, channels, height, width)
        # time: (batch_size, 1)
        # text: (batch_size, seq_len, text_embed_dim)
        # vec: (batch_size, vector_embed_dim)
        # mask: (batch_size, num_tokens)
        batch_size, channels, height, width = img.shape

        # Reshape and transmute img to have shape (batch_size, height*width, channels)
        img = img.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)

        time = self.time_embedder(time)
        
        # Vector embedding (timestep + vector_embeddings)
        vec = self.vector_embedder(vec) + time  # (batch_size, embed_dim)

        # Image embedding
        img = self.image_embedder(img)

        # (height, width, embed_dim)
        sincos_pos_embed = sincos_2d(self.embed_dim, height, width)
        sincos_pos_embed = sincos_pos_embed.to(img.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        img = img + sincos_pos_embed

        # Caption embedding
        txt = self.text_embedder(txt)  # (batch_size, embed_dim)

        # Patch-mixer
        img, txt = self.token_mixer(img, txt, vec, height, width)

        # Remove masked patches
        if mask is not None:
            img = remove_masked_tokens(img, mask)

        # Backbone transformer model
        img = self.backbone(img, txt, vec, mask, height, width)
        
        # Final output layer
        # (bs, unmasked_num_tokens, embed_dim) -> (bs, unmasked_num_tokens, in_channels)
        img = self.output_layer(img, vec)

        # Add masked patches
        if mask is not None:
            # (bs, unmasked_num_tokens, in_channels) -> (bs, num_tokens, in_channels)
            img = add_masked_tokens(img, mask)

        img = img.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width).contiguous()
        
        return img
    
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