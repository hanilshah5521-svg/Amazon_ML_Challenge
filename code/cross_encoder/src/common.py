"""Paths shared by the cross-encoder scripts. Run from code/cross_encoder/src/."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]           # .../Amazon_ML_Challenge-0.967
DATA = ROOT / "student_resource" / "dataset"
WORK = ROOT / "work"
CE = WORK / "ce"
CE.mkdir(parents=True, exist_ok=True)
MODEL_NAME = os.environ.get("CE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MAX_LEN = int(os.environ.get("CE_MAX_LEN", 96))
