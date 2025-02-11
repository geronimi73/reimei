import math
import torch.nn as nn
from torch.nn.modules.normalization import RMSNorm
from transformer.moedit import MoeMLP
from .embed import PatchEmbed, sincos_2d, TimestepEmbedder, MLPEmbedder, OutputLayer
from .utils import remove_masked_tokens, add_masked_tokens, unpatchify
from .backbone import BackboneParams, TransformerBackbone
from .token_mixer import TokenMixer
import torch
from config import AE_SCALING_FACTOR
from dataclasses import dataclass

@dataclass
class ReiMeiParameters:
    channels: int
    patch_size: tuple[int, int]
    embed_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    siglip_dim: int
    bert_dim: int
    num_experts: int = 4
    capacity_factor: float = 2.0
    shared_experts: int = 2
    dropout: float = 0.1
    token_mixer_layers: int = 2
    image_text_expert_ratio: int = 4
    # m_d: float = 1.0

class ReiMei(nn.Module):
    """
    ReiMei is a image diffusion transformer model.

        Args:
        channels (int): Number of input channels in the image data.
        patch_size (Tuple[int, int]): Size of the patch.
        embed_dim (int): Dimension of the embedding space.
        num_layers (int): Number of layers in the transformer backbone.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        mlp_dim (int): Dimension of the multi-layer perceptron.
        text_embed_dim (int): Dimension of the text embedding.
        vector_embed_dim (int): Dimension of the vector embedding.
        num_experts (int, optional): Number of experts in the transformer backbone. Default is 4.
        capacity_factor (float, optional): Average number of experts per token. Default is 2.0.
        shared_experts (int, optional): Number of shared experts in the transformer backbone. Default is 2.
        dropout (float, optional): Dropout rate. Default is 0.1.
        patch_mixer_layers (int, optional): Number of layers in the patch mixer. Default is 2.

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
    def __init__(self, params: ReiMeiParameters):
        super().__init__()
        
        self.embed_dim = params.embed_dim
        self.pos_emb_dim = params.embed_dim // params.num_heads
        self.channels = params.channels
        self.patch_size = params.patch_size
        
        # Timestep embedding
        self.time_embedder = TimestepEmbedder(self.embed_dim)

        # Image embedding
        self.image_embedder = PatchEmbed(self.channels, self.embed_dim, self.patch_size)
        
        # Text embedding
        self.siglip_embedder = MLPEmbedder(params.siglip_dim, self.embed_dim, hidden_dim=self.embed_dim*4, num_layers=4)
        self.bert_embedder = MLPEmbedder(params.bert_dim, self.embed_dim, hidden_dim=self.embed_dim*4, num_layers=4)

        # Only bert needs normalization, siglip embeddings are in reasonable range
        self.bert_norm = RMSNorm(params.bert_dim)

        # Vector (y) embedding
        self.vector_embedder = MLPEmbedder(params.siglip_dim + params.bert_dim, self.embed_dim, hidden_dim=self.embed_dim*4, num_layers=4)
        
        # TokenMixer
        self.token_mixer = TokenMixer(self.embed_dim, params.num_heads, params.token_mixer_layers, num_experts=params.num_experts, capacity_factor=params.capacity_factor, exp_ratio=params.image_text_expert_ratio)

        backbone_params = BackboneParams(
            input_dim=self.channels,
            embed_dim=self.embed_dim,
            num_layers=params.num_layers,
            num_heads=params.num_heads,
            mlp_dim=params.mlp_dim,
            num_experts=params.num_experts,
            capacity_factor=params.capacity_factor,
            shared_experts=params.shared_experts,
            dropout=params.dropout,
            image_text_expert_ratio=params.image_text_expert_ratio,
        )

        # Backbone transformer model
        self.backbone = TransformerBackbone(backbone_params)
        
        self.output_layer = OutputLayer(self.embed_dim, self.channels * self.patch_size[0] * self.patch_size[1])

        self.initialize_weights()

    def initialize_weights(self):
        s = 1.0 / math.sqrt(self.embed_dim)

        # Initialize all linear layers and biases
        # def _basic_init(module):
        #     if isinstance(module, nn.LayerNorm):
        #         if module.weight is not None:
        #             nn.init.constant_(module.weight, 1.0)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)
        #     elif isinstance(module, nn.Linear):
        #         nn.init.normal_(module.weight, std=s)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)

        # Initialize all linear layers and biases
        # def _basic_init(module):
        #     if isinstance(module, nn.Linear):
        #         nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)
        #     elif isinstance(module, nn.Conv2d):
        #         # Initialize convolutional layers like linear layers
        #         nn.init.xavier_uniform_(module.weight.view(module.weight.size(0), -1))
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)
        #     elif isinstance(module, nn.LayerNorm):
        #         # Initialize LayerNorm layers
        #         if module.weight is not None:
        #             nn.init.constant_(module.weight, 1.0)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)


        # # Apply basic initialization to all modules
        # self.apply(_basic_init)

        # def _mlp_embedder_init(module):
        #     if isinstance(module, MLPEmbedder):
        #         # First linear layer: use Kaiming uniform for layers followed by GELU.
        #         nn.init.xavier_uniform_(module.mlp[0].weight)
        #         if module.mlp[0].bias is not None:
        #             nn.init.constant_(module.mlp[0].bias, 0)
        #         # Second linear layer: use Xavier uniform.
        #         nn.init.xavier_uniform_(module.mlp[2].weight)
        #         if module.mlp[2].bias is not None:
        #             nn.init.constant_(module.mlp[2].bias, 0)

        # self.apply(_mlp_embedder_init)

        # def _moe_mlp_init(module):
        #     if isinstance(module, MoeMLP):
        #         nn.init.xavier_uniform_(module.gate_proj.weight)
        #         nn.init.xavier_uniform_(module.up_proj.weight)
        #         nn.init.xavier_uniform_(module.down_proj.weight)

        # self.apply(_moe_mlp_init)

        # Zero-out the last linear layer in the output to ensure initial predictions are zero
        nn.init.constant_(self.output_layer.mlp.mlp[-1].weight, 0)
        nn.init.constant_(self.output_layer.mlp.mlp[-1].bias, 0)

    def forward(self, img, time, sig_txt, sig_vec, bert_txt, bert_vec, mask=None):
        # img: (batch_size, channels, height, width)
        # time: (batch_size, 1)
        # sig_txt: (batch_size, seq_len, siglip_dim)
        # sig_vec: (batch_size, siglip_dim)
        # bert_txt: (batch_size, seq_len, bert_dim)
        # bert_vec: (batch_size, bert_dim)
        # mask: (batch_size, num_tokens)
        batch_size, channels, height, width = img.shape
        ps_h, ps_w = self.patch_size
        patched_h, patched_w = height // ps_h, width // ps_w

        # Text embeddings
        sig_txt = self.siglip_embedder(sig_txt)
        bert_txt = self.bert_embedder(self.bert_norm(bert_txt))
        txt = torch.cat([sig_txt, bert_txt], dim=1)
        
        # Vector embedding (timestep + vector_embeddings)
        time = self.time_embedder(time)

        vec = torch.cat([sig_vec, self.bert_norm(bert_vec)], dim=1)
        vec = self.vector_embedder(vec) + time  # (batch_size, embed_dim)

        # Image embedding
        img = self.image_embedder(img)

        # (height // patch_size_h, width // patch_size_w, embed_dim)
        sincos_pos_embed = sincos_2d(self.embed_dim, patched_h, patched_w)
        sincos_pos_embed = sincos_pos_embed.to(device=img.device, dtype=img.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        
        img = img + sincos_pos_embed

        # Patch-mixer
        img, txt = self.token_mixer(img, txt, vec, patched_h, patched_w)
        # img = self.token_mixer(img)

        # Remove masked patches
        if mask is not None:
            img = remove_masked_tokens(img, mask)

        # Backbone transformer model
        img = self.backbone(img, txt, vec, mask, patched_h, patched_w)
        
        # Final output layer
        # (bs, unmasked_num_tokens, embed_dim) -> (bs, unmasked_num_tokens, in_channels)
        img = self.output_layer(img, vec)

        # Add masked patches
        if mask is not None:
            # (bs, unmasked_num_tokens, in_channels) -> (bs, num_tokens, in_channels)
            img = add_masked_tokens(img, mask)

        # img = img.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width).contiguous()
        img = unpatchify(img, self.patch_size, height, width)
        
        return img
    
    @torch.no_grad()
    def sample(self, z, sig_emb, sig_vec, bert_emb, bert_vec, sample_steps=50, cfg=3.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device, torch.bfloat16).view([b, *([1] * len(z.shape[1:]))])
        images = [z]

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device, torch.bfloat16)

            vc = self(z, t, sig_emb, sig_vec, bert_emb, bert_vec, None).to(torch.bfloat16)
            if cfg != 1.0:
                null_sig_emb = torch.zeros_like(sig_emb)
                null_sig_vec = torch.zeros_like(sig_vec)
                null_bert_emb = torch.zeros_like(bert_emb)
                null_bert_vec = torch.zeros_like(bert_vec)
                vu = self(z, t, null_sig_emb, null_sig_vec, null_bert_emb, null_bert_vec, None)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)

        # print("Std dev and mean of sampled images", torch.std_mean(images[-1]))

        return (images[-1] / AE_SCALING_FACTOR)
