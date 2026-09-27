"""Generic scorer: any parquet of (s1, cand [, text_s1, text_cand]) pairs -> parquet (s1, cand, p_ce).
This is the integration point: point it at any pipeline's candidate file and get back one probability per pair.

Usage: python score_pairs.py <in.parquet> <out.parquet> [--model-dir DIR] [--batch-size N]
If the input lacks text_s1/text_cand, pass --split train|test and --source-dir to fetch raw text (see prep_data.py).
"""
import argparse
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from common import CE, MAX_LEN
from train import PairDataset, collate
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def score(df, model_dir, batch_size=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    df = df.with_columns(pl.lit(0.0).alias("y")).sort(pl.col("text_s1").str.len_chars())  # length-sort: faster padding
    dl = DataLoader(PairDataset(df), batch_size=batch_size, collate_fn=collate(tok))
    out = np.empty(df.height, dtype=np.float32)
    t0 = time.time()
    i = 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        for batch in dl:
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.squeeze(-1).float().cpu().numpy()
            out[i:i + len(logits)] = 1 / (1 + np.exp(-logits))
            i += len(logits)
            if i % (batch_size * 100) < batch_size:
                print(f"   [{time.time() - t0:.0f}s] scored {i:,}/{df.height:,}", flush=True)
    return df.with_columns(pl.Series("p_ce", out)).select("s1", "cand", "p_ce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path"); ap.add_argument("out_path")
    ap.add_argument("--model-dir", default=str(CE / "model"))
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()
    df = pl.read_parquet(args.in_path)
    assert {"s1", "cand", "text_s1", "text_cand"} <= set(df.columns), \
        "input needs s1, cand, text_s1, text_cand (see prep_data.py for how to build these from raw source TSVs)"
    n_in = df.height
    res = score(df, args.model_dir, args.batch_size)
    assert res.height == n_in and res.select(["s1", "cand"]).n_unique() == n_in, "row count / uniqueness mismatch"
    res.write_parquet(args.out_path)
    print(f"wrote {args.out_path}: {res.height:,} rows", flush=True)


if __name__ == "__main__":
    main()
