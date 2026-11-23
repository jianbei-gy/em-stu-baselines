#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.
"""Fashion-MNIST small-CNN baseline. Metric: official test top-1 accuracy."""

from __future__ import annotations

import argparse
import json
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
from torch.utils.data import DataLoader
from torchvision import transforms

from model.cnn import SmallCNN


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


def link_or_use(cache_subdir: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = Path("/workspace/starter_baselines/.cache/torchvision") / cache_subdir
    dest = DATA_DIR / cache_subdir
    if dest.exists():
        return DATA_DIR
    if cache.exists():
        dest.symlink_to(cache, target_is_directory=True)
    return DATA_DIR


def load_datasets():
    from torchvision.datasets import FashionMNIST
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    root = link_or_use("FashionMNIST")
    return (
        FashionMNIST(root, train=True, download=True, transform=tfm),
        FashionMNIST(root, train=False, download=True, transform=tfm),
    )


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
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
    model = SmallCNN(1, 10).to(device)
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
        "project_id": "cv_fashion_mnist_cnn",
        "task": "image classification",
        "dataset": "Fashion-MNIST (official, torchvision)",
        "model": "SmallCNN",
        "metrics": {"top1_accuracy": history[-1]["test_acc"], "best_top1_accuracy": best_acc},
        "split": "official test (10,000 images)",
        "epochs": args.epochs,
        "device": str(device),
        "seed": args.seed,
        "duration_sec": round(elapsed, 2),
        "timestamp": now_shanghai(),
        "command": f"python scripts/train.py --epochs {args.epochs} --batch-size {args.batch_size} --lr {args.lr} --seed {args.seed}",
        "history": history,
        "notes": "Trained on official 60,000 training images; reported metric is official test top-1 accuracy.",
    }
    write_results(args.output, payload)
    print(json.dumps(payload["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
