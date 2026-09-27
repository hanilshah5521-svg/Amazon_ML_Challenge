"""Stage and upload the exported features as PRIVATE Kaggle datasets (no data is copied: hard links on E:).

  python kaggle_upload.py train   # work_hv/feat_train/part_*.parquet            -> <user>/amlc26-feat-train
  python kaggle_upload.py test    # work_hv/feat_test_<country>/part_*.parquet   -> <user>/amlc26-feat-test
                                  #   staged flat as <country>_part_*.parquet
Needs C:/Users/Hanil/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create New Token).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
KAGGLE = [sys.executable, "-c", "from kaggle.cli import main; main()"]


def user():
    return json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]


def stage(split):
    dst = WORK / f"kaggle_{split}"
    dst.mkdir(exist_ok=True)
    src = [("", WORK / "feat_train")] if split == "train" else \
          [(f"{c}_", WORK / f"feat_test_{c}") for c in ("France", "India", "US")]
    n = 0
    for prefix, d in src:
        for p in sorted(d.glob("part_*.parquet")):
            link = dst / f"{prefix}{p.name}"
            if not link.exists():
                os.link(p, link)
            n += 1
    meta = {"title": f"amlc26 feat {split}", "id": f"{user()}/amlc26-feat-{split}", "licenses": [{"name": "CC0-1.0"}]}
    (dst / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    size = sum(p.stat().st_size for p in dst.glob("*.parquet")) / 1e9
    print(f"staged {n} parts ({size:.2f} GB) in {dst}", flush=True)
    return dst


def main():
    """python kaggle_upload.py <train|test> [version "<message>"]  (second form uploads a NEW VERSION of the dataset;
    for re-exported parts, delete their old links in kaggle_<split>/ first, because stage() keeps existing links)."""
    dst = stage(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "version":
        msg = sys.argv[3] if len(sys.argv) > 3 else "update"
        subprocess.run(KAGGLE + ["datasets", "version", "-p", str(dst), "-m", msg, "-q"], check=True)
    else:
        # private by default (no --public); -q: no per-file progress bars in the log
        subprocess.run(KAGGLE + ["datasets", "create", "-p", str(dst), "-q"], check=True)
    print("upload done", flush=True)


if __name__ == "__main__":
    main()
