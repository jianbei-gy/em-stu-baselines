#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.
"""CIFAR-10 ResNet-18 (CIFAR stem) baseline. Metric: official test top-1 accuracy."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

def _project_root() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    here = Path(__file__).resolve()
    # local layout: <proj>/scripts/train.py
    if here.parent.name == "scripts":
        return here.parents[1]
    return here.parent

PROJ = _project_root()
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

DATA_DIR = PROJ / "data"
CKPT_DIR = PROJ / "model" / "checkpoints"
RESULTS_PATH = PROJ / "results.json"


def now_shanghai() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_results(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text)
    cwd_copy = Path.cwd() / "results.json"
    if cwd_copy.resolve() != path.resolve():
        cwd_copy.write_text(text)
    print("Wrote", path, flush=True)

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from model.resnet import cifar_resnet18


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


class HFImageDataset(Dataset):
    def __init__(self, split, transform, image_keys=("img", "image"), label_keys=("label", "fine_label")):
        self.split = split
        self.transform = transform
        row0 = split[0]
        self.image_key = next(k for k in image_keys if k in row0)
        self.label_key = next(k for k in label_keys if k in row0)

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        row = self.split[int(idx)]
        img = row[self.image_key]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        if img.mode != "RGB" and 3 == 3:
            img = img.convert("RGB")
        elif img.mode != "L" and 3 == 1:
            img = img.convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row[self.label_key])


def load_datasets():
    """Official CIFAR-10 (50k/10k). Prefer torchvision files in ./data; else HF uoft-cs/cifar10 mirror."""
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.environ.setdefault("HF_HOME", "/workspace/starter_baselines/.cache/huggingface")
        from datasets import load_dataset
        raw = load_dataset("uoft-cs/cifar10")
        print("loaded CIFAR-10 via Hugging Face official mirror uoft-cs/cifar10", flush=True)
        return HFImageDataset(raw["train"], train_tfm), HFImageDataset(raw["test"], test_tfm)
    except Exception as e:
        print("HF CIFAR-10 failed (%s); trying torchvision" % e, flush=True)
    from torchvision.datasets import CIFAR10
    train_ds = CIFAR10(DATA_DIR, train=True, download=True, transform=train_tfm)
    test_ds = CIFAR10(DATA_DIR, train=False, download=True, transform=test_tfm)
    print("loaded CIFAR-10 via torchvision ->", DATA_DIR, flush=True)
    return train_ds, test_ds


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output", type=Path, default=RESULTS_PATH)
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device()
    print("device", device, flush=True)
    train_ds, test_ds = load_datasets()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)
    model = cifar_resnet18(10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    history = []
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, device, True)
        te_loss, te_acc = run_epoch(model, test_loader, None, device, False)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "test_loss": te_loss, "test_acc": te_acc})
        print(f"epoch {epoch}/{args.epochs} train_acc={tr_acc:.4f} test_acc={te_acc:.4f} loss={tr_loss:.4f}", flush=True)
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "test_acc": te_acc}, CKPT_DIR / "best.pt")
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "test_acc": history[-1]["test_acc"]}, CKPT_DIR / "last.pt")
    elapsed = time.time() - t0
    payload = {
        "project_id": "cv_cifar10_resnet",
        "task": "image classification",
        "dataset": "CIFAR-10 (official 50k/10k; torchvision or HF uoft-cs/cifar10 mirror)",
        "model": "ResNet-18 (CIFAR stem)",
        "metrics": {"top1_accuracy": history[-1]["test_acc"], "best_top1_accuracy": best_acc},
        "split": "official test (10,000 images)",
        "epochs": args.epochs,
        "device": str(device),
        "seed": args.seed,
        "duration_sec": round(elapsed, 2),
        "timestamp": now_shanghai(),
        "command": f"python scripts/train.py --epochs {args.epochs} --batch-size {args.batch_size} --lr {args.lr} --seed {args.seed}",
        "history": history,
        "notes": "Official complete CIFAR-10 train/test. ResNet-18 uses 3x3 stride-1 conv1 and no first max-pool. A few epochs on GPU; not a SOTA run.",
    }
    write_results(args.output, payload)
    print(json.dumps(payload["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
