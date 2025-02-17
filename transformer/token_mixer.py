from dataclasses import dataclass
import torch
import torch.nn as nn
from transformer.moedit import DoubleStreamBlock, DiTBlock

@dataclass
class TokenMixerParameters:
    use_mmdit: bool = True
    use_ec: bool = False
    embed_dim: int = 1152
    num_heads: int = 1152 // 64
    num_layers: int = 2
    mlp_ratio: int = 4
    num_experts: int = 8
    capacity_factor: int = 2.0
    pretraining_tp: int = 2
    num_shared_experts: int = 2
    exp_ratio: int = 4

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
        params: TokenMixerParameters,
    ):
        super().__init__()
        if params.use_mmdit:
            self.layers = nn.ModuleList([
                DoubleStreamBlock(
                    hidden_size=params.embed_dim,
                    num_heads=params.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    num_experts=params.num_experts,
                    capacity_factor=params.capacity_factor,
                    pretraining_tp=params.pretraining_tp,
                    num_shared_experts=params.num_shared_experts,
                    exp_ratio=params.exp_ratio,
                    use_expert_choice=params.use_ec,
                )
                for _ in range(params.num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                DiTBlock(
                    hidden_size=params.embed_dim,
                    num_heads=params.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    num_experts=params.num_experts,
                    num_experts_per_tok=params.capacity_factor,
                    pretraining_tp=params.pretraining_tp,
                    num_shared_experts=params.num_shared_experts,
                    attn_drop=0.1
                )
                for _ in range(params.num_layers)
            ])



    def forward(
        self,
        img: torch.Tensor,       # [B, L_img, embed_dim]
        txt: torch.Tensor,       # [B, L_txt, embed_dim]
        vec: torch.Tensor,       # [B, embed_dim]
        img_rope: torch.Tensor,
        txt_rope: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Passes (img, txt) through each DiTBlock in sequence. Each DiTBlock performs
        cross-attention over the concatenation of (img, txt), plus MoE-based feedforwards.
        """
        for layer in self.layers:
            img, txt = layer(img, txt, vec, img_rope, txt_rope)
            # img = layer(img, vec)
        return img, txt
        # return img