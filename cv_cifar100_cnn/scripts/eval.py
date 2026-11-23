#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.
"""Evaluate a saved checkpoint on the official test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

CKPT_DIR = PROJ / "model" / "checkpoints"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


from torch.utils.data import DataLoader

# Reuse dataset loader from train
import importlib.util
_spec = importlib.util.spec_from_file_location("train_mod", Path(__file__).with_name("train.py"))
train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_mod)
from model.cnn import SmallCNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=CKPT_DIR / "best.pt")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()
    device = get_device()
    _, test_ds = train_mod.load_datasets()
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = SmallCNN(3, 100).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / max(total, 1)
    print(json.dumps({"ckpt": str(args.ckpt), "split": "official test", "top1_accuracy": acc, "n": total}, indent=2))


if __name__ == "__main__":
    main()
