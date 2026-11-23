#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.
"""SASRec on official MovieLens 100K. Metrics: full-rank leave-one-out HR@10 and NDCG@10."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def _project_root() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    here = Path(__file__).resolve()
    if here.parent.name == "scripts":
        return here.parents[1]
    return here.parent


PROJ = _project_root()
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

DATA_DIR = PROJ / "data"
CKPT_DIR = PROJ / "model" / "checkpoints"
RESULTS_PATH = PROJ / "results.json"
CACHE_DATASETS = Path("/workspace/starter_baselines/.cache/datasets")


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


def leave_one_out(by_user: dict) -> tuple[list, list, list, int]:
    """by_user: raw_user -> list[(timestamp, raw_item)]. Returns train/val/test lists and n_items."""
    cleaned = {}
    for u, evs in by_user.items():
        evs = sorted(evs, key=lambda x: (x[0], str(x[1])))
        dedup = []
        seen = set()
        for t, i in reversed(evs):
            if i in seen:
                continue
            seen.add(i)
            dedup.append((t, i))
        evs = list(reversed(dedup))
        if len(evs) >= 3:
            cleaned[u] = evs
    item_ids = sorted({i for evs in cleaned.values() for _, i in evs}, key=str)
    item2idx = {i: idx + 1 for idx, i in enumerate(item_ids)}  # 0 = pad
    users = sorted(cleaned, key=str)
    train, val, test = [], [], []
    for u in users:
        items = [item2idx[i] for _, i in cleaned[u]]
        train.append(items[:-2])
        val.append(items[-2])
        test.append(items[-1])
    return train, val, test, len(item_ids), users, item_ids


def save_splits(path: Path, train, val, test, n_items, users, items, extra: dict) -> dict:
    stats = {
        "n_users": len(train),
        "n_items": n_items,
        "n_train_interactions": int(sum(len(s) for s in train)),
        "n_val": len(val),
        "n_test": len(test),
        "min_train_len": int(min(len(s) for s in train)),
        "max_train_len": int(max(len(s) for s in train)),
        "median_train_len": float(np.median([len(s) for s in train])),
        **extra,
    }
    payload = {
        "n_users": len(train),
        "n_items": n_items,
        "train": train,
        "val": val,
        "test": test,
        "user_raw": [str(u) for u in users],
        "item_raw": [str(i) for i in items],
        "stats": stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    (path.parent / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print("saved splits", path, stats, flush=True)
    return payload


def load_splits(path: Path) -> dict:
    return json.loads(path.read_text())


def pack_left(seq: list[int], max_len: int) -> list[int]:
    seq = list(seq[-max_len:])
    return [0] * (max_len - len(seq)) + seq


@torch.no_grad()
def full_rank_metrics(scores: torch.Tensor, target: torch.Tensor, exclude: torch.Tensor, k: int = 10):
    """scores [B, n_items+1], target [B], exclude [B, n_items+1] bool (True = do not rank)."""
    if not torch.isfinite(scores).all():
        raise ValueError("Cannot rank non-finite recommendation scores")
    scores = scores.clone()
    scores[exclude] = -1e9
    scores[:, 0] = -1e9
    target_score = scores.gather(1, target.view(-1, 1))
    rank = (scores > target_score).sum(dim=1) + 1
    hit = (rank <= k).float()
    ndcg = torch.where(rank <= k, 1.0 / torch.log2(rank.float() + 1.0), torch.zeros_like(rank, dtype=torch.float))
    return {
        "hr": float(hit.mean().item()),
        "ndcg": float(ndcg.mean().item()),
        "mean_rank": float(rank.float().mean().item()),
        "n": int(target.numel()),
    }


class SeqEvalDataset(Dataset):
    def __init__(self, train_seqs, val_items, test_items, max_len: int, split: str):
        self.max_len = max_len
        self.split = split
        self.rows = []
        for u, hist in enumerate(train_seqs):
            if split == "val":
                self.rows.append((hist, val_items[u], hist))
            else:
                self.rows.append((hist + [val_items[u]], test_items[u], hist + [val_items[u]]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        hist, target, seen = self.rows[idx]
        seq = torch.tensor(pack_left(hist, self.max_len), dtype=torch.long)
        seen_t = torch.tensor(seen, dtype=torch.long)
        return seq, torch.tensor(target, dtype=torch.long), seen_t


def collate_eval(batch, n_items: int):
    seqs, targets, seens = zip(*batch)
    seqs = torch.stack(seqs, 0)
    targets = torch.stack(targets, 0)
    exclude = torch.zeros(len(batch), n_items + 1, dtype=torch.bool)
    for i, s in enumerate(seens):
        exclude[i, s] = True
    return seqs, targets, exclude


class SASRec(nn.Module):
    def __init__(self, n_items: int, hidden: int = 64, max_len: int = 50, n_layers: int = 2, n_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_items = n_items
        self.hidden = hidden
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.emb_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.item_emb.weight.data[0].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        b, t = seq.shape
        pos = torch.arange(t, device=seq.device).unsqueeze(0).expand(b, t)
        x = self.item_emb(seq) * math.sqrt(self.hidden) + self.pos_emb(pos)
        x = self.emb_drop(x)
        pad_mask = seq == 0
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=seq.device), diagonal=1)
        # Padded queries can attend to themselves; valid queries still ignore padding.
        # An entirely masked query would produce NaN in the evaluation attention kernel.
        blocked = causal.unsqueeze(0) | pad_mask.unsqueeze(1)
        blocked.diagonal(dim1=-2, dim2=-1).fill_(False)
        heads = self.encoder.layers[0].self_attn.num_heads
        x = self.encoder(x, mask=blocked.repeat_interleave(heads, dim=0))
        x = self.norm(x)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x

    def all_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.item_emb.weight.t()


class SASRecTrainDataset(Dataset):
    def __init__(self, train_seqs, max_len: int):
        self.max_len = max_len
        self.inputs = []
        self.targets = []
        for seq in train_seqs:
            if len(seq) < 1:
                continue
            src = seq[:-1] if len(seq) >= 2 else seq
            tgt = seq[1:] if len(seq) >= 2 else seq
            # If only one train item, still allow predicting it from a short context of itself? skip if <2
            if len(seq) < 2:
                continue
            self.inputs.append(torch.tensor(pack_left(src, max_len), dtype=torch.long))
            self.targets.append(torch.tensor(pack_left(tgt, max_len), dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def prepare_ml100k() -> dict:
    raw_cache = CACHE_DATASETS / "ml100k" / "ml-100k" / "u.data"
    raw_local = DATA_DIR / "u.data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not raw_local.exists() and raw_cache.exists():
        try:
            raw_local.symlink_to(raw_cache)
        except FileExistsError:
            pass
    src = raw_local if raw_local.exists() else raw_cache
    if not src.exists():
        raise FileNotFoundError("MovieLens 100K u.data not found")
    splits_path = DATA_DIR / "processed" / "splits.json"
    cache_splits = CACHE_DATASETS / "ml100k" / "leave_one_out" / "splits.json"
    if splits_path.exists():
        return load_splits(splits_path)
    if cache_splits.exists():
        DATA_DIR.joinpath("processed").mkdir(parents=True, exist_ok=True)
        if not splits_path.exists():
            splits_path.symlink_to(cache_splits)
            stats_src = cache_splits.with_name("stats.json")
            stats_dst = splits_path.with_name("stats.json")
            if stats_src.exists() and not stats_dst.exists():
                stats_dst.symlink_to(stats_src)
        return load_splits(cache_splits)
    by_user: dict = {}
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        u, i, r, t = line.split()
        by_user.setdefault(int(u), []).append((int(t), int(i)))
    train, val, test, n_items, users, items = leave_one_out(by_user)
    extra = {
        "dataset": "MovieLens 100K (GroupLens official u.data)",
        "source": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "protocol": "time-sorted leave-one-out; implicit (all ratings); users with <3 interactions dropped",
    }
    return save_splits(splits_path, train, val, test, n_items, users, items, extra)


def load_checkpoint(path, device):
    # Earlier checkpoints saved an output path using Python's version-specific pathlib name.
    path_types = [(PurePosixPath, name) for name in ("pathlib.PosixPath", "pathlib._local.PosixPath")]
    with torch.serialization.safe_globals(path_types):
        return torch.load(path, map_location=device, weights_only=True)


def ensure_data() -> dict:
    return prepare_ml100k()


@torch.no_grad()
def evaluate_split(model, splits, max_len, device, split: str, batch_size: int = 256):
    n_items = splits["n_items"]
    ds = SeqEvalDataset(splits["train"], splits["val"], splits["test"], max_len, split)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_eval(b, n_items),
    )
    hrs, ndcgs, n = 0.0, 0.0, 0
    model.eval()
    for seq, target, exclude in loader:
        seq = seq.to(device)
        target = target.to(device)
        exclude = exclude.to(device)
        hidden = model(seq)[:, -1, :]
        logits = model.all_logits(hidden)
        m = full_rank_metrics(logits, target, exclude, k=10)
        hrs += m["hr"] * m["n"]
        ndcgs += m["ndcg"] * m["n"]
        n += m["n"]
    return {"hr@10": hrs / max(n, 1), "ndcg@10": ndcgs / max(n, 1), "n": n}


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total, n_tok = 0.0, 0, 0
    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad(set_to_none=True)
        hidden = model(src)
        logits = model.all_logits(hidden)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=0,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        valid = int((tgt != 0).sum().item())
        total_loss += float(loss.item()) * valid
        n_tok += valid
        total += src.size(0)
    return total_loss / max(n_tok, 1)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--max-seconds", type=int, default=10000)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--output", type=Path, default=RESULTS_PATH)
    return p.parse_args()


def main():
    args = get_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    device = get_device()
    splits = ensure_data()
    n_items = splits["n_items"]
    model = SASRec(
        n_items,
        hidden=args.hidden,
        max_len=args.max_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)
    train_ds = SASRecTrainDataset(splits["train"], args.max_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_ndcg = -1.0
    best_path = CKPT_DIR / "best.pt"
    history = []
    t0 = time.time()
    stopped = "completed"
    bad = 0
    for epoch in range(1, args.epochs + 1):
        if time.time() - t0 > args.max_seconds:
            stopped = "max_seconds"
            break
        tr_loss = train_one_epoch(model, loader, optimizer, device)
        val = evaluate_split(model, splits, args.max_len, device, "val", args.eval_batch_size)
        row = {"epoch": epoch, "train_loss": tr_loss, "val_hr@10": val["hr@10"], "val_ndcg@10": val["ndcg@10"]}
        history.append(row)
        print(row, flush=True)
        if val["ndcg@10"] > best_ndcg:
            best_ndcg = val["ndcg@10"]
            torch.save(
                {"model": model.state_dict(), "n_items": n_items, "args": {**vars(args), "output": str(args.output)}, "val": val, "epoch": epoch},
                best_path,
            )
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                stopped = "early_stop"
                break
    ckpt = load_checkpoint(best_path, device) if best_path.exists() else None
    if ckpt:
        model.load_state_dict(ckpt["model"])
        best_epoch = ckpt["epoch"]
    else:
        best_epoch = 0
    test = evaluate_split(model, splits, args.max_len, device, "test", args.eval_batch_size)
    payload = {
        "project_id": "gr_ml100k_sasrec",
        "task": "sequential recommendation",
        "dataset": "MovieLens 100K (GroupLens official u.data, time-sorted leave-one-out implicit)",
        "model": "SASRec",
        "metrics": {"hr@10": test["hr@10"], "ndcg@10": test["ndcg@10"]},
        "split": "leave-one-out last item (full ranking, excluding train+val items)",
        "n_users": splits["n_users"],
        "n_items": splits["n_items"],
        "epochs": best_epoch,
        "device": str(device),
        "seed": args.seed,
        "duration_sec": round(time.time() - t0, 2),
        "timestamp": now_shanghai(),
        "command": "python scripts/train.py " + " ".join([
            f"--epochs {args.epochs}",
            f"--batch-size {args.batch_size}",
            f"--lr {args.lr}",
            f"--hidden {args.hidden}",
            f"--max-len {args.max_len}",
            f"--seed {args.seed}",
        ]),
        "history": history,
        "stop_reason": stopped,
        "notes": "Causal SASRec (hidden 64, max len 50, 2 layers). Metrics are full-rank HR@10 and NDCG@10 on the official GroupLens MovieLens 100K interactions after time-sorted leave-one-out. Numbers come from this run only.",
        "stats": splits.get("stats", {}),
    }
    write_results(args.output, payload)
    print(json.dumps({"test": test, "best_epoch": best_epoch, "stop_reason": stopped}, indent=2), flush=True)


if __name__ == "__main__":
    main()
