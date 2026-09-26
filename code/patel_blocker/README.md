# patel blocker: second candidate source for the learned cut (submission 5)

An S2/S3 → S1 blocker whose candidates are **added to** the Harsh V per-state TF-IDF top-40 before the learned cut. K stays 8. This folder only produces the candidate files that `business_entity_resolution/src/run_patel.sh` reads through `OUR_CANDS_DIR`.

How it works:
- **Normalisation** (`normalize.py`, `prep.py`):
  - Indic → Latin transliteration: one offset table for all 8 Brahmic scripts.
  - NFKD accent stripping and domain-name stems.
  - **S2/S3 → S1 token substitutions learned from train true pairs of half the S1 entities** (even id-bucket, `--subs learn-half`), e.g. praivet→private, rd→road, texas→tx. Address pairs must look like abbreviations.
  - Test uses the train map (`--subs use`); no test data is used to learn anything.
- **Retrieval** (`block.py`): an IDF inverted index per country over:
  - name and address unigrams and adjacent-token bigrams
  - compact-name and core-compact-name keys (frequent tokens dropped: df > 0.05% of the country's S1)
- **Deep re-rank:**
  - Records whose best IDF score is < 52 get their top-200 re-ranked by name token-set + address token-set + IDF/5, keeping 8.
  - Confident records keep 3.
  - Candidates below 0.8 × the best re-rank score are dropped.
  - Output per pair: `rank`, `rr` (re-rank score).
- **Recall** of true pairs on all train: 0.9824 overall (India 0.980, US 0.984), rank-1 0.9685, 8.5 candidates per S1.

## Run (Python 3.10+; `pip install -r requirements.txt`)
```bash
cd code/patel_blocker
python src/prep.py  --split train --subs learn-half      # ~1 min
python src/block.py --split train                        # ~30 min, 16 GB RAM
python src/prep.py  --split test  --subs use             # ~1 min
python src/block.py --split test                         # ~25 min
python src/recall_report.py                              # optional: candidate recall by segment
# then: OUR_CANDS_DIR=$(pwd)/../../work/patel_blocker bash ../business_entity_resolution/src/run_patel.sh
```
- Outputs go to `work/patel_blocker/{train,test}/{records,cands}.parquet` in the repo root (git-ignored).
- This code version breaks ranking ties deterministically, sorting by (score, S1 row id). The run that produced submission 5 used an earlier build that broke ties by row order. Rebuilt candidates can therefore differ in a few tied pairs.
