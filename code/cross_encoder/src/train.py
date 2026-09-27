"""Fine-tune a small MIT/Apache-2.0 multilingual cross-encoder to score (S1, candidate) pairs.

Trains ONLY on pairs whose S1 is not in the held-out set (work/ce/train_pairs.parquet, from prep_data.py).
Time-budgeted: stops after --minutes of wall-clock training (saves the checkpoint so far) so it never overruns
the deadline; a full epoch is the default target if the budget allows it.

Usage: python train.py [--n-s1 N] [--minutes M] [--batch-size B] [--smoke-test]
"""
import argparse
import time

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import CE, MAX_LEN, MODEL_NAME


class PairDataset(Dataset):
    def __init__(self, df):
        self.a = df["text_s1"].to_list()
        self.b = df["text_cand"].to_list()
        self.y = df["y"].to_numpy().astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i]


def collate(tok):
    def f(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc
    return f


def auc_on(model, tok, df, device, n=20000, batch_size=256):
    df = df.sample(min(n, df.height), seed=0) if df.height > n else df
    dl = DataLoader(PairDataset(df), batch_size=batch_size, collate_fn=collate(tok))
    model.eval()
    probs, ys = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        for batch in dl:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.squeeze(-1).float().cpu()
            probs.append(torch.sigmoid(logits).numpy())
            ys.append(labels.numpy())
    model.train()
    y, p = np.concatenate(ys), np.concatenate(probs)
    return roc_auc_score(y, p) if len(set(y.tolist())) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=250_000, help="max distinct S1 records to train on")
    ap.add_argument("--minutes", type=float, default=75, help="training wall-clock budget")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--smoke-test", action="store_true", help="tiny run to check the pipeline works end to end")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={MODEL_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)

    df = pl.read_parquet(CE / "train_pairs.parquet")
    s1 = df["s1"].unique().sample(min(args.n_s1, df["s1"].n_unique()), seed=0)
    df = df.filter(pl.col("s1").is_in(s1)).sample(fraction=1.0, seed=1)  # shuffle
    val = pl.read_parquet(CE / "val_pairs.parquet")
    if args.smoke_test:
        df, val, args.minutes = df.head(2000), val.head(2000), 5
    print(f"training on {df.height:,} pairs from {df['s1'].n_unique():,} S1 ({df['y'].mean():.1%} positive)", flush=True)

    dl = DataLoader(PairDataset(df), batch_size=args.batch_size, shuffle=False, collate_fn=collate(tok))
    steps = len(dl)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    t0 = time.time()
    budget = args.minutes * 60
    eval_every = max(1, steps // 5)
    model.train()
    for i, batch in enumerate(dl):
        if time.time() - t0 > budget:
            print(f"[{time.time() - t0:.0f}s] time budget reached at step {i}/{steps}, stopping early", flush=True)
            break
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if i % eval_every == 0 or i == steps - 1:
            auc = auc_on(model, tok, val, device)
            print(f"[{time.time() - t0:6.0f}s] step {i:>6}/{steps} loss {loss.item():.4f} held-out AUC {auc:.4f}", flush=True)

    out = CE / "model"
    out.mkdir(exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    final_auc = auc_on(model, tok, val, device, n=200_000)
    print(f"DONE trained {time.time() - t0:.0f}s, final held-out AUC (up to 200k pairs) = {final_auc:.4f}", flush=True)
    (CE / "train_meta.txt").write_text(
        f"model={MODEL_NAME}\nmax_len={MAX_LEN}\nn_s1={args.n_s1}\nlr={args.lr}\nbatch_size={args.batch_size}\n"
        f"held_out_auc={final_auc:.4f}\ntrain_seconds={time.time() - t0:.0f}\ntorch={torch.__version__}\n")


if __name__ == "__main__":
    main()
