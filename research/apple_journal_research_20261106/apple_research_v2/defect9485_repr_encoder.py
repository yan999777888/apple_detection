"""Local forward of Meta DINOv2 ViT-S/14; tensor-only official checkpoint."""

import torch
from torch import nn
from torch.nn import functional as F


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(384))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(384, eps=1e-6)
        self.norm2 = nn.LayerNorm(384, eps=1e-6)
        self.attn = nn.ModuleDict(
            dict(qkv=nn.Linear(384, 3 * 384), proj=nn.Linear(384, 384))
        )
        self.mlp = nn.ModuleDict(
            dict(fc1=nn.Linear(384, 1536), fc2=nn.Linear(1536, 384))
        )
        self.ls1 = Scale()
        self.ls2 = Scale()

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = (
            self.attn["qkv"](self.norm1(x))
            .view(b, n, 3, 6, 64)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        residual = (
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        )
        x = x + self.ls1.gamma * self.attn["proj"](residual)
        return x + self.ls2.gamma * self.mlp["fc2"](
            F.gelu(self.mlp["fc1"](self.norm2(x)))
        )


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, 384))
        self.mask_token = nn.Parameter(torch.zeros(1, 384))
        self.patch_embed = nn.ModuleDict(dict(proj=nn.Conv2d(3, 384, 14, stride=14)))
        self.blocks = nn.ModuleList(Block() for _ in range(12))
        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.inference_mode()
    def forward(self, crops):
        x = self.patch_embed["proj"](
            (crops.permute(0, 3, 1, 2).float() / 255 - self.rgb_mean) / self.rgb_std
        )
        side = x.shape[-1]
        spatial = self.pos_embed[:, 1:].reshape(1, 37, 37, 384).permute(0, 3, 1, 2)
        pos = F.interpolate(
            spatial,
            scale_factor=((side + 0.1) / 37,) * 2,
            mode="bicubic",
            align_corners=False,
        )
        x = torch.cat(
            [
                self.cls_token.expand(len(x), -1, -1) + self.pos_embed[:, :1],
                (x + pos).flatten(2).transpose(1, 2),
            ],
            1,
        )
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in (5, 11):
                z = self.norm(x) if i == 11 else F.layer_norm(x, (384,))
                tiles = z[:, 1:].reshape(len(x), side, side, 384)
                center = tiles[:, 1:-1, 1:-1].mean((1, 2))
                features.extend([z[:, 0], center, center - tiles.mean((1, 2))])
        return torch.cat(features, 1)
