# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.

import numpy as np
import torch
import torch.nn as nn

class FactorizationMachine(nn.Module):
    """Standard FM: linear + pairwise interactions via the square-sum trick.

    y = w0 + sum_i w_i x_i + 0.5 * ( (sum_i v_i x_i)^2 - sum_i (v_i x_i)^2 )
    Inputs are one index per field (all features treated as categorical).
    """
    def __init__(self, field_dims, embed_dim=16):
        super().__init__()
        offsets = np.cumsum([0] + list(field_dims)[:-1]).astype(np.int64)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(int(sum(field_dims)), 1)
        self.emb = nn.Embedding(int(sum(field_dims)), embed_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x):
        x = x + self.offsets
        lin = self.bias + self.linear(x).sum(dim=1).squeeze(-1)
        v = self.emb(x)
        square_of_sum = v.sum(dim=1) ** 2
        sum_of_square = (v ** 2).sum(dim=1)
        inter = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)
        return lin + inter
