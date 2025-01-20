
import torch
import torch.nn as nn
from transformer.moedit import DoubleStreamBlock

class TokenMixer(nn.Module):
    """
    Each layer expects:
        - img:       [B, L_img, embed_dim]
        - txt:       [B, L_txt, embed_dim]
        - vec:       [B, embed_dim]            (conditioning vector for Modulation)
        - h          Height of the original image
        - w          Width of the original image
    and returns the updated (img, txt) after `num_layers` of DoubleStreamBlock.
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
            DoubleStreamBlock(
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

    def forward(
        self,
        img: torch.Tensor,       # [B, L_img, embed_dim]
        txt: torch.Tensor,       # [B, L_txt, embed_dim]
        vec: torch.Tensor,       # [B, embed_dim]
        h: int,
        w: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Passes (img, txt) through each DiTBlock in sequence. Each DiTBlock performs
        cross-attention over the concatenation of (img, txt), plus MoE-based feedforwards.
        """
        for layer in self.layers:
            img, txt = layer(img, txt, vec, None, h, w)
        return img, txt