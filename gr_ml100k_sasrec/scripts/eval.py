#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.
"""Evaluate a saved sequential-rec checkpoint with full-rank HR@10 and NDCG@10."""
import argparse
import importlib.util
import json
from pathlib import Path

here = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("train_mod", here.with_name("train.py"))
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=train_mod.CKPT_DIR / "best.pt")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()
    device = train_mod.get_device()
    splits = train_mod.ensure_data()
    ckpt = train_mod.load_checkpoint(args.ckpt, device)
    hidden = ckpt.get("args", {}).get("hidden", 64)
    max_len = ckpt.get("args", {}).get("max_len", 50)
    n_layers = ckpt.get("args", {}).get("n_layers", 2)
    n_heads = ckpt.get("args", {}).get("n_heads", 2)
    dropout = ckpt.get("args", {}).get("dropout", 0.2)
    n_items = ckpt.get("n_items", splits["n_items"])
    model_cls = getattr(train_mod, "SASRec", None) or getattr(train_mod, "BERT4Rec")
    model = model_cls(n_items, hidden=hidden, max_len=max_len, n_layers=n_layers, n_heads=n_heads, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model"])
    metrics = train_mod.evaluate_split(model, splits, max_len, device, args.split, args.batch_size)
    print(json.dumps({"ckpt": str(args.ckpt), "split": args.split, **metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
