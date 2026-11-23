# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.

"""Small CNN used by several CV starter projects."""
from __future__ import annotations

import torch.nn as nn


class SmallCNN(nn.Module):
    """Five 3x3 conv layers with BatchNorm + ReLU, three max-pools, AdaptiveAvgPool, MLP head.

    Layout:
      Conv(in, 32, 3, pad=1) - BN - ReLU
      Conv(32, 32, 3, pad=1) - BN - ReLU
      MaxPool(2)
      Conv(32, 64, 3, pad=1) - BN - ReLU
      Conv(64, 64, 3, pad=1) - BN - ReLU
      MaxPool(2)
      Conv(64, 128, 3, pad=1) - BN - ReLU
      MaxPool(2)
      AdaptiveAvgPool2d(1)
      Flatten - Linear(128, 256) - ReLU - Dropout(0.3) - Linear(256, num_classes)
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
