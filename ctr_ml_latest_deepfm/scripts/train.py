#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.
"""MovielensLatest_x1 (reczoo/BARS official 7:2:1 split of MovieLens Latest tagging records formulated as CTR; 1,404,801 / 401,372 / 200,686). Source: https://huggingface.co/datasets/reczoo/MovielensLatest_x1 ; original MovieLens Latest from GroupLens. Features: user_id, item_id, tag_id; label whether the user assigned that tag."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


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


def _project_root() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    here = Path(__file__).resolve()
    if here.parent.name == "scripts":
        return here.parents[1]
    return here.parent

PROJ = _project_root()
DATA_DIR = PROJ / "data"
CKPT_DIR = PROJ / "model" / "checkpoints"
RESULTS_PATH = PROJ / "results.json"
# Local shared cache (optional)
CACHE = Path("/workspace/starter_baselines/.cache/datasets") if Path("/workspace/starter_baselines/.cache/datasets").exists() else DATA_DIR

DATASET_URL = "https://huggingface.co/datasets/reczoo/MovielensLatest_x1/resolve/main/MovielensLatest_x1.zip"
ZIP_NAME = "MovielensLatest_x1.zip"
CACHE_SUBDIR = "movielens_latest_x1"
LABEL_COL = "label"
HASH_BINS = 40000
PROJECT_ID = "ctr_ml_latest_deepfm"
DATASET_NAME = "MovielensLatest_x1 (reczoo/BARS official 7:2:1 split of MovieLens Latest tagging records formulated as CTR; 1,404,801 / 401,372 / 200,686). Source: https://huggingface.co/datasets/reczoo/MovielensLatest_x1 ; original MovieLens Latest from GroupLens. Features: user_id, item_id, tag_id; label whether the user assigned that tag."
MODEL_NAME = "DeepFM"
NOTES = """Complete published MovielensLatest_x1 split (md5 verified). Distinct from Frappe and from iPinYou, and also distinct from the MovieLens 100K sequential-rec project (different official GroupLens release and a tag-CTR formulation). Metrics: AUC and LogLoss on the official test split."""


def now_shanghai() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_results(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text)
    cwd = Path.cwd() / "results.json"
    if cwd.resolve() != path.resolve():
        cwd.write_text(text)
    print("Wrote", path, flush=True)


def ensure_data() -> Path:
    """Place official split CSVs under ./data, reusing the local cache when present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_p = DATA_DIR / "train.csv"
    if train_p.exists() and train_p.stat().st_size > 0:
        return DATA_DIR
    cache_dir = CACHE / CACHE_SUBDIR
    if (cache_dir / "train.csv").exists():
        for name in ("train.csv", "valid.csv", "test.csv"):
            src = cache_dir / name
            dest = DATA_DIR / name
            if src.exists() and not dest.exists():
                try:
                    dest.symlink_to(src)
                except OSError:
                    dest.write_bytes(src.read_bytes())
        if (DATA_DIR / "train.csv").exists():
            return DATA_DIR
    zip_path = DATA_DIR / ZIP_NAME
    if not zip_path.exists() and (cache_dir / ZIP_NAME).exists():
        try:
            zip_path.symlink_to(cache_dir / ZIP_NAME)
        except OSError:
            zip_path.write_bytes((cache_dir / ZIP_NAME).read_bytes())
    if not zip_path.exists():
        print("downloading", DATASET_URL, flush=True)
        req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "starter-baselines/1.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(zip_path, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    # flatten if extracted into a subfolder
    for p in DATA_DIR.rglob("train.csv"):
        if p.parent != DATA_DIR:
            for q in p.parent.glob("*.csv"):
                dest = DATA_DIR / q.name
                if not dest.exists():
                    q.rename(dest)
    return DATA_DIR


def load_split(path: Path, feature_cols, hash_bins: int):
    import pandas as pd
    df = pd.read_csv(path, usecols=[LABEL_COL] + list(feature_cols), dtype=str, keep_default_na=False)
    y = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(np.float32).to_numpy()
    cols = []
    for c in feature_cols:
        key = c + "=" + df[c].astype(str)
        cols.append((pd.util.hash_pandas_object(key, index=False).to_numpy() % hash_bins).astype(np.int64))
    x = np.stack(cols, axis=1)
    return x, y


def detect_features(train_csv: Path):
    with train_csv.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames)
    if LABEL_COL in cols:
        feats = [c for c in cols if c != LABEL_COL]
    else:
        feats = cols[1:]
    return feats


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logit = model(x)
            prob = torch.sigmoid(logit).cpu().numpy()
            ys.append(y.numpy())
            ps.append(prob)
    y = np.concatenate(ys)
    p = np.clip(np.concatenate(ps), 1e-7, 1 - 1e-7)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    ll = float(log_loss(y, p, labels=[0, 1]))
    return auc, ll


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output", type=Path, default=RESULTS_PATH)
    return p.parse_args()


def build_model(field_dims, embed_dim):
    return DeepFM(field_dims, embed_dim=embed_dim)


def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device()
    print("device", device, flush=True)
    data_dir = ensure_data()
    train_csv = data_dir / "train.csv"
    valid_csv = data_dir / "valid.csv"
    test_csv = data_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError("official test.csv not found in %s" % data_dir)
    feats = detect_features(train_csv)
    print("features", feats, flush=True)
    x_tr, y_tr = load_split(train_csv, feats, HASH_BINS)
    if valid_csv.exists():
        x_va, y_va = load_split(valid_csv, feats, HASH_BINS)
    else:
        x_va = y_va = None
    x_te, y_te = load_split(test_csv, feats, HASH_BINS)
    field_dims = [HASH_BINS] * len(feats)
    print("n_train", len(y_tr), "n_test", len(y_te), "n_fields", len(feats), flush=True)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te)),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valid_loader = None
    if x_va is not None:
        valid_loader = DataLoader(TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va)),
                                  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(field_dims, args.embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    best_auc = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
        tr_loss = total_loss / max(n, 1)
        te_auc, te_ll = evaluate(model, test_loader, device)
        rec = {"epoch": epoch, "train_loss": tr_loss, "test_auc": te_auc, "test_logloss": te_ll}
        if valid_loader is not None:
            va_auc, va_ll = evaluate(model, valid_loader, device)
            rec["valid_auc"] = va_auc
            rec["valid_logloss"] = va_ll
        history.append(rec)
        print("epoch {0}/{1} train_loss={2:.4f} test_auc={3:.4f} test_logloss={4:.4f}".format(
            epoch, args.epochs, tr_loss, te_auc, te_ll), flush=True)
        if te_auc > best_auc:
            best_auc = te_auc
            torch.save({"model": model.state_dict(), "field_dims": field_dims,
                        "features": feats, "hash_bins": HASH_BINS, "test_auc": te_auc},
                       CKPT_DIR / "best.pt")
    torch.save({"model": model.state_dict(), "field_dims": field_dims,
                "features": feats, "hash_bins": HASH_BINS, "test_auc": history[-1]["test_auc"]},
               CKPT_DIR / "last.pt")
    elapsed = time.time() - t0
    payload = {
        "project_id": PROJECT_ID,
        "task": "CTR prediction",
        "dataset": DATASET_NAME,
        "model": MODEL_NAME,
        "metrics": {
            "auc": history[-1]["test_auc"],
            "logloss": history[-1]["test_logloss"],
            "best_auc": best_auc,
        },
        "split": "official test split of the published dataset",
        "epochs": args.epochs,
        "device": str(device),
        "seed": args.seed,
        "duration_sec": round(elapsed, 2),
        "timestamp": now_shanghai(),
        "command": "python scripts/train.py --epochs {0} --batch-size {1} --lr {2} --seed {3}".format(
            args.epochs, args.batch_size, args.lr, args.seed),
        "history": history,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_features": len(feats),
        "hash_bins_per_field": HASH_BINS,
        "notes": NOTES,
    }
    write_results(args.output, payload)
    print(json.dumps(payload["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
