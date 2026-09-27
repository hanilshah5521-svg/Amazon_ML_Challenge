# Cross-encoder pair scorer

A second opinion on candidate pairs the main pipeline is uncertain about: a small MIT/Apache-2.0 multilingual
transformer, fine-tuned on `(business_name | business_address)` text pairs, that outputs one probability per pair.
Designed to blend with any other pipeline's scores by a simple join on `(s1, cand)`.

**Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, ~118M parameters), loaded as
`AutoModelForSequenceClassification(num_labels=1)`. Fine-tuned from scratch on the competition's own training data —
no external data, no lookups.

**Built on:** a laptop with an 8 GB RTX 4060 (not Kaggle GPU), against this repo's own K=8 candidate cut
(`work/{val,extra,extra2}_candidates_cut.parquet`, `work/test_candidates_cut_{France,India,US}.parquet`), since this
machine has no Kaggle credentials for `hanilshah/amlc26-feat-{train,test}`. Entity IDs are the same raw dataset IDs
everywhere, so `p_ce` joins onto any other pipeline's candidate pairs — but only pairs that overlap with *this*
K=8 cut get a score; a pair present only in another pipeline's candidate set will have no `p_ce` after the join.

## Files
| File | Role |
|---|---|
| `src/common.py` | Paths, model name, max sequence length |
| `src/prep_data.py` | Builds text pairs + labels from the raw source TSVs and this repo's candidate-cut parquet files |
| `src/train.py` | Fine-tunes the cross-encoder (time-budgeted; trains only on non-held-out S1) |
| `src/score_pairs.py` | **The integration point.** Any parquet with `s1, cand, text_s1, text_cand` in → `s1, cand, p_ce` out |
| `requirements.txt` | Pinned versions |

## Reproduce
```bash
pip install -r requirements.txt
cd src
python prep_data.py                          # -> work/ce/{train,val}_pairs.parquet, test_pairs_{country}.parquet
python train.py --n-s1 250000 --minutes 75    # -> work/ce/model/, work/ce/train_meta.txt
python score_pairs.py ../../../work/ce/val_pairs.parquet          ../../../work/ce/ce_val.parquet
python score_pairs.py ../../../work/ce/test_pairs_France.parquet  ../../../work/ce/ce_test_France.parquet
python score_pairs.py ../../../work/ce/test_pairs_India.parquet   ../../../work/ce/ce_test_India.parquet
python score_pairs.py ../../../work/ce/test_pairs_US.parquet      ../../../work/ce/ce_test_US.parquet
```

## Output contract
Every scored file: `s1` (str), `cand` (str), `p_ce` (float32 probability), one row per input pair, no duplicates,
no missing pairs. `score_pairs.py` asserts this before writing.

## To blend with another pipeline's scores
```python
blended = own_scores.join(ce_scores, on=["s1", "cand"], how="left")  # p_ce is null where no CE score exists
blended["p_final"] = w * blended["p_ours"] + (1 - w) * blended["p_ce"].fill_null(blended["p_ours"])
```
Tune `w` on held-out half A, measure macro F0.5 on half B, keep only if it wins — same protocol as the rest of this
repo's experiments (`documentation/experiments.md`).

## Known limitations
- Trained and validated on India/US only (France has no labels); untested on France beyond being scored by the same model.
- Candidate universe is this repo's own K=8 cut, not necessarily identical to another pipeline's — see the coverage note above.
- Time-budgeted to fit a single-GPU laptop session; `train_meta.txt` (written after training) records the actual
  training set size, steps completed and held-out AUC.
