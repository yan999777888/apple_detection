"""Experimental visible-to-amodal residual; effectiveness needs ablation.

Uses predicted visibility only. No labels, shape templates or fabricated
MinneApple amodal targets enter the model. Zero initialization preserves the
parent logits exactly before the first optimization step.
"""
from math import sqrt

import torch
from torch import nn


class VisibilityGatedAmodalResidual(nn.Module):
    def __init__(self, feature_dim=256, bottleneck=64):
        super().__init__()
        # Optional adaptation must not consume the baseline's random stream.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            self.residual_embedder = nn.Sequential(nn.Linear(feature_dim, bottleneck),
                                                   nn.GELU(), nn.Linear(bottleneck, feature_dim))
            self.query_gate = nn.Linear(feature_dim, 1)
            nn.init.zeros_(self.residual_embedder[-1].weight)
            nn.init.zeros_(self.residual_embedder[-1].bias)
            nn.init.zeros_(self.query_gate.weight)
            nn.init.constant_(self.query_gate.bias, -2.0)
        self.feature_dim = feature_dim

    def forward(self, queries, pixels, visible_logits, amodal_logits):
        # Keep A-specific corrections from directly rewriting V logits.
        hidden_support = (1.0 - visible_logits.sigmoid()).detach()
        residual = torch.einsum("bqc,bchw->bqhw", self.residual_embedder(queries), pixels)
        gate = self.query_gate(queries).sigmoid().unsqueeze(-1)
        return amodal_logits + hidden_support * gate * residual / sqrt(self.feature_dim)
