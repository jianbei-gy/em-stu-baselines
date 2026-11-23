# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.

"""CIFAR-adapted ResNet-18: 3x3 stride-1 conv1, no first max-pool."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def cifar_resnet18(num_classes: int) -> nn.Module:
    """Standard ResNet-18 body with a CIFAR stem.

    Changes from ImageNet ResNet-18:
      - conv1: 7x7 s=2, p=3  ->  3x3 s=1, p=1, bias=False
      - maxpool: 3x3 s=2     ->  Identity (keep 32x32 until layer2)
      - fc: Linear(512, num_classes)
    Remaining blocks are the usual 2-2-2-2 BasicBlocks (64-128-256-512 channels).
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
