
import torch
import torch.nn as nn
from transformer.moedit import DoubleStreamBlock, DiTBlock

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
        capacity_factor: int = 2.0,
        pretraining_tp: int = 2,
        num_shared_experts: int = 2,
        exp_ratio: int = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DoubleStreamBlock(
                hidden_size=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
                pretraining_tp=pretraining_tp,
                num_shared_experts=num_shared_experts,
                exp_ratio=exp_ratio
            )
            # DiTBlock(
            #     hidden_size=embed_dim,
            #     num_heads=num_heads,
            #     mlp_ratio=mlp_ratio,
            #     num_experts=num_experts,
            #     num_experts_per_tok=capacity_factor,
            #     pretraining_tp=pretraining_tp,
            #     num_shared_experts=num_shared_experts,
            #     attn_drop=0.1
            # )
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
            # img = layer(img, vec)
        return img, txt
        # return img