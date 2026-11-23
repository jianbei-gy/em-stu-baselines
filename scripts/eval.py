#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.
"""Evaluate a saved CTR checkpoint on the official test split."""
import importlib.util
import json
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

here = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("train_mod", here.with_name("train.py"))
train_mod = importlib.util.module_from_spec(spec)
# Avoid re-running main
spec.loader.exec_module(train_mod)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=train_mod.CKPT_DIR / "best.pt")
    p.add_argument("--batch-size", type=int, default=4096)
    args = p.parse_args()
    device = train_mod.get_device()
    data_dir = train_mod.ensure_data()
    feats = train_mod.detect_features(data_dir / "train.csv")
    x_te, y_te = train_mod.load_split(data_dir / "test.csv", feats, train_mod.HASH_BINS)
    ckpt = torch.load(args.ckpt, map_location=device)
    field_dims = ckpt.get("field_dims", [train_mod.HASH_BINS] * len(feats))
    embed_dim = ckpt["model"]["emb.weight"].shape[1]
    model = train_mod.build_model(field_dims, embed_dim).to(device)
    model.load_state_dict(ckpt["model"])
    loader = DataLoader(TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te)),
                        batch_size=args.batch_size, shuffle=False)
    auc, ll = train_mod.evaluate(model, loader, device)
    print(json.dumps({"ckpt": str(args.ckpt), "auc": auc, "logloss": ll, "n": int(len(y_te))}, indent=2))


if __name__ == "__main__":
    main()
