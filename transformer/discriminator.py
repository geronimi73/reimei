import math
import torch.nn as nn
from transformer.math import rope_ids
from .embed import EmbedND, PatchEmbed, TimestepEmbedder, MLPEmbedder, OutputLayer
from .utils import remove_masked_tokens
from .backbone import BackboneParams, TransformerBackbone
import torch
from dataclasses import dataclass

@dataclass
class DiscriminatorParameters:
    use_mmdit: bool = True
    use_ec: bool = False
    use_moe: bool = False
    channels: int = 32
    patch_size: tuple[int, int] = (1,1)
    embed_dim: int = 1152
    num_layers: int = 24
    num_heads: int = 1152 // 64
    siglip_dim: int = 1152
    num_experts: int = 4
    capacity_factor: float = 2.0
    shared_experts: int = 2
    dropout: float = 0.1
    image_text_expert_ratio: int = 4
    # m_d: float = 1.0

class Discriminator(nn.Module):
    """
    A GAN setup's Discriminator

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

    Attributes:
        embed_dim (int): Dimension of the embedding space.
        channels (int): Number of input channels in the image data.
        time_embedder (TimestepEmbedder): Timestep embedding layer.
        image_embedder (MLPEmbedder): Image embedding layer.
        text_embedder (MLPEmbedder): Text embedding layer.
        vector_embedder (MLPEmbedder): Vector embedding layer.
        backbone (TransformerBackbone): Transformer backbone model.
        output (MLPEmbedder): Output layer.
    """
    def __init__(self, params: DiscriminatorParameters):
        super().__init__()
        self.params = params
        self.embed_dim = params.embed_dim
        self.head_dim = params.embed_dim // params.num_heads
        self.channels = params.channels
        self.patch_size = params.patch_size
        self.use_mmdit = params.use_mmdit
            
        # Image embedding
        self.image_embedder = PatchEmbed(self.channels, self.embed_dim, self.patch_size)
        
        # Text embedding
        self.siglip_embedder = MLPEmbedder(params.siglip_dim, self.embed_dim, hidden_dim=self.embed_dim*4, num_layers=1)
    
        # Vector (y) embedding
        self.vector_embedder = MLPEmbedder(params.siglip_dim, self.embed_dim, hidden_dim=self.embed_dim*4, num_layers=2)

        self.rope_embedder = EmbedND(dim=self.head_dim)

        # Backbone transformer model
        backbone_params = BackboneParams(
            use_mmdit=params.use_mmdit,
            use_ec=params.use_ec,
            use_moe=params.use_moe,
            embed_dim=self.embed_dim,
            num_layers=params.num_layers,
            num_heads=params.num_heads,
            num_experts=params.num_experts,
            capacity_factor=params.capacity_factor,
            shared_experts=params.shared_experts,
            dropout=params.dropout,
            image_text_expert_ratio=params.image_text_expert_ratio,
        )
        self.backbone = TransformerBackbone(backbone_params)
        
        self.output_layer = OutputLayer(self.embed_dim, 1)

        self.initialize_weights()

    def initialize_weights(self):
        s = 1.0 / math.sqrt(self.embed_dim)

        # # Initialize all linear layers and biases
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

        # # Initialize all linear layers and biases
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

        # Zero-out the last linear layer in the output to ensure initial predictions are zero
        nn.init.constant_(self.output_layer.mlp.mlp[-1].weight, 0)
        nn.init.constant_(self.output_layer.mlp.mlp[-1].bias, 0)

    def forward(self, img, txt, vec, img_mask=None, txt_mask=None):
        # img: (batch_size, channels, height, width)
        # txt: (batch_size, seq_len, siglip_dim)
        # vec: (batch_size, siglip_dim)
        # mask: (batch_size, num_tokens)
        batch_size, channels, height, width = img.shape
        ps_h, ps_w = self.patch_size
        patched_h, patched_w = height // ps_h, width // ps_w

        # Text embeddings
        txt = self.siglip_embedder(txt)

        _, seq_len, _ = txt.shape

        vec = self.vector_embedder(vec)

        # Image embedding
        img = self.image_embedder(img)

        rope_id = rope_ids(batch_size, patched_h, patched_w, seq_len, img.device, img.dtype)
        rope_pe = self.rope_embedder(torch.cat(rope_id, dim=1))

        # Remove masked patches
        txt_rope_id, img_rope_id = rope_id
        if img_mask is not None:
            img = remove_masked_tokens(img, img_mask)
            img_rope_id = remove_masked_tokens(img_rope_id, img_mask)

        if txt_mask is not None:
            txt = remove_masked_tokens(txt, txt_mask)
            txt_rope_id = remove_masked_tokens(txt_rope_id, txt_mask)

        rope_pe = self.rope_embedder(torch.cat((txt_rope_id, img_rope_id), dim=1))

        # Backbone transformer model
        img = self.backbone(img, txt, vec, rope_pe)

        # Final output layer
        # (bs, unmasked_num_tokens, embed_dim) -> (bs, unmasked_num_tokens, 1)
        img = self.output_layer(img, vec)

        # (bs, unmasked_num_tokens, 1) -> (bs, 1)
        logit = img.mean(dim=1)
        return logit
    

def approximate_r1_loss(
    discriminator,
    latents,
    siglip_emb,
    siglip_vec,
    img_mask,
    txt_mask,
    sigma=0.01,
    Lambda=100.0,
):
    """
    Approx. R1 via finite differences:
        L_{aR1} = || D(x) - D(x + noise) ||^2
    """
    noise = sigma * torch.randn_like(latents)
    d_real = discriminator(latents, siglip_emb, siglip_vec, img_mask, txt_mask)
    d_noisy = discriminator(latents + noise, siglip_emb, siglip_vec, img_mask, txt_mask)
    return ((d_real - d_noisy).pow(2).mean()) * Lambda


def approximate_r2_loss(
    discriminator,
    fake_images,
    siglip_emb,
    siglip_vec,
    img_mask,
    txt_mask,
    sigma=0.01,
    Lambda=100.0,

):
    """
    Approx. R2 via finite differences:
        L_{aR2} = || D(x_fake) - D(x_fake + noise) ||^2
    """
    noise = sigma * torch.randn_like(fake_images)
    d_fake = discriminator(fake_images, siglip_emb, siglip_vec, img_mask, txt_mask)
    d_fake_noisy = discriminator(fake_images + noise, siglip_emb, siglip_vec, img_mask, txt_mask)
    return ((d_fake - d_fake_noisy).pow(2).mean()) * Lambda

def gan_loss_with_approximate_penalties(
    discriminator,
    generator,
    latents,
    x_t,
    t,
    siglip_emb,
    siglip_vec,
    img_mask,
    txt_mask,
    discriminator_turn=True,
    sigma=0.01,
    Lambda=100.0
):
    """
    Non saturating Relativistic GAN loss of the form:
        E_{z,x}[ f(-(D(G(z)) - D(x))) ].
    for the discriminator, and form:
        E_{z,x}[ f(-(D(x) - D(G(z)))) ].
    for the generator.

    Adds approximate R1 and R2 penalties to the discriminator loss.

    Args:
        discriminator: Discriminator network D
        generator: Generator network G
        real_images: A batch of real data (x)
        z: Noise tensor sampled from p_z
        discriminator_turn: If True, calculates loss for the discriminator, else for the generator
        f: Callable for f(D(fake) - D(real)) [defaults to torch.nn.functional.softplus]
        generator_args: Extra positional args for G
        generator_kwargs: Extra keyword args for G
        disc_args: Extra positional args for D
        disc_kwargs: Extra keyword args for D
        sigma: Standard deviation of the noise added to the real images
        Lambda: Weight for the approximate R1 and R2 penalties
    """
    # Default the function to logistic_f if none is provided

    f = torch.nn.functional.softplus

    # Generate fake images
    fake_v = generator(x_t, t, siglip_emb, siglip_vec, img_mask, txt_mask)
    texp = t.view(-1, 1, 1, 1)
    fake_images = x_t - (fake_v * texp)

    # Evaluate discriminator
    disc_real = discriminator(latents, siglip_emb, siglip_vec, img_mask, txt_mask)
    disc_fake = discriminator(fake_images, siglip_emb, siglip_vec, img_mask, txt_mask)

    # Compute the loss using default or provided f
    if discriminator_turn:
        loss = f(disc_fake - disc_real).mean()
        loss += approximate_r1_loss(discriminator, latents, siglip_emb, siglip_vec, img_mask, txt_mask)
        loss += approximate_r2_loss(discriminator, fake_images, siglip_emb, siglip_vec, img_mask, txt_mask)
    else:
        loss = f(disc_real - disc_fake)
    return loss, fake_v