# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.

import numpy as np
import torch
import torch.nn as nn

class DeepFM(nn.Module):
    """DeepFM (Guo et al., IJCAI 2017).

    Shared categorical embeddings feed:
      1) a linear + FM pairwise term (same as Factorization Machine);
      2) a 2-layer MLP on the concatenated field embeddings.
    The logit is the sum of the three parts.
    """
    def __init__(self, field_dims, embed_dim=16, mlp_dims=(128, 64), dropout=0.2):
        super().__init__()
        offsets = np.cumsum([0] + list(field_dims)[:-1]).astype(np.int64)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(int(sum(field_dims)), 1)
        self.emb = nn.Embedding(int(sum(field_dims)), embed_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.emb.weight)
        layers = []
        in_dim = embed_dim * len(field_dims)
        for h in mlp_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.offsets
        lin = self.bias + self.linear(x).sum(dim=1).squeeze(-1)
        v = self.emb(x)
        square_of_sum = v.sum(dim=1) ** 2
        sum_of_square = (v ** 2).sum(dim=1)
        inter = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)
        deep = self.mlp(v.reshape(v.size(0), -1)).squeeze(1)
        return lin + inter + deep
