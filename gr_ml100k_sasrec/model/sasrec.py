#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.
from __future__ import annotations
import math
import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, n_items: int, hidden: int = 64, max_len: int = 50, n_layers: int = 2, n_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_items = n_items
        self.hidden = hidden
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.emb_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.item_emb.weight.data[0].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        b, t = seq.shape
        pos = torch.arange(t, device=seq.device).unsqueeze(0).expand(b, t)
        x = self.item_emb(seq) * math.sqrt(self.hidden) + self.pos_emb(pos)
        x = self.emb_drop(x)
        pad_mask = seq == 0
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=seq.device), diagonal=1)
        # Padded queries can attend to themselves; valid queries still ignore padding.
        # An entirely masked query would produce NaN in the evaluation attention kernel.
        blocked = causal.unsqueeze(0) | pad_mask.unsqueeze(1)
        blocked.diagonal(dim1=-2, dim2=-1).fill_(False)
        heads = self.encoder.layers[0].self_attn.num_heads
        x = self.encoder(x, mask=blocked.repeat_interleave(heads, dim=0))
        x = self.norm(x)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x

    def all_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.item_emb.weight.t()
