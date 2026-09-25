# Amazon ML Challenge 2026: Business Entity Resolution

For every business record in **Source 1** (the deduplicated reference), find all records in
**Source 2** and **Source 3** that refer to the same real-world business. Records have a name, an
address and a country (US and India in train; France appears only in test). Scoring is
**macro F0.5** per Source 1 record: precision counts twice as much as recall, and a record with no
true match scores 1.0 only if nothing is predicted.

**Current leaderboard score: 0.915** (LightGBM matcher, 25 Sep 2026).

## Pipeline

```
raw TSVs ─▶ normalize ─▶ candidate search (blocking) ─▶ pair features ─▶ LightGBM ─▶ decisions ─▶ TSV outputs
```

1. **Normalization** (`normalize.py`)
   - Indian-script names (Hindi, Telugu, Tamil, Kannada, Bengali, Gujarati, Punjabi, Malayalam,
     Odia) become English through a word dictionary **learned from the training pairs**. It covers
     96% of test Indic tokens; unidecode is the fallback.
   - Accents are removed; text is lowercased; punctuation becomes spaces; dotted abbreviations
     are joined (`L.L.C.` → `llc`).
   - Legal forms are canonicalized (Private Limited / Pvt Ltd, SARL, LLC…).
   - Alias markers are handled: in `X f/k/a Y` and `X doing business as Y`, Y is kept as the
     entity's own name.
   - Look-alike digits are fixed (`Pub1ications`, `5ARL`).
   - Addresses: USPS and French street abbreviations, state/region canonicalization
     (TG ≡ Telangana ≡ తెలంగాణ, Nord ≡ Hauts-de-France), Indian old/new city names, ordinals
     and number words, Roman numerals, and filler words (Unit, Apt, PMB, N°…) are handled.
   - Three name views: `name`, `name_core` (legal forms and stopwords removed) and `name_key`
     (generic business words also removed).
2. **Candidate search** (`blocking.py`), run per country:
   - top-20 by character 3-gram TF-IDF on names (spaces removed)
   - top-20 by word TF-IDF on addresses
   - the two lists are merged and re-ranked, using `sparse_dot_topn` for sparse top-K search
3. **Pair features** (`pair_features.py`, `lgbm_matcher.py`):
   - string similarities on name, core and key
   - house-number agreement
   - legal-form conflict
   - empty-address flags
   - comparisons against the other candidates of the same Source 1 record (gap to best, rank,
     number of close candidates)
4. **Matcher** (`lgbm_matcher.py`): LightGBM, validated with 5-fold cross-validation grouped by
   Source 1 record. A candidate is accepted if its probability clears the per-country threshold,
   and each Source 2/3 record goes only to its best Source 1 record.
5. **Output**: `output/matching_results.tsv` and `output/candidate_pairs.tsv`, checked with the
   organisers' `validate_submission.py`.

## Repository layout

| Path | Contents |
|---|---|
| `code/business_entity_resolution/src/common.py` | Paths and TSV/ground-truth loaders |
| `…/src/normalize.py` | All normalization rules; builds the Indic dictionary; writes normalized parquet files |
| `…/src/blocking.py` | Candidate search: `eval` (recall on held-out train) and `test` modes |
| `…/src/pair_features.py` | Legal-form and house-number checks |
| `…/src/lgbm_matcher.py` | LightGBM training, cross-validated evaluation, `submit` mode |
| `…/src/evaluation.py` | Macro F0.5 scorer (README formula), match decision rule, submission TSV writer |
| `…/src/eda.py`, `edge_miner.py`, `france_miner.py`, `france_patterns.py`, `explore_neighbors.py`, `blocking_analysis.py` | Analysis only: where the rules came from. Not needed to reproduce outputs. |
| `research_notes/` | Literature and competition research (blocking, matchers, normalization, benchmarks, industry systems) |
| `phase-1.ipynb`, `Shreya nb - 2.ipynb` | Team exploration notebooks |

## How to run

Put the competition data at `student_resource/dataset/{train,test}/` (it's git-ignored).
Intermediate files go to `work/` and outputs to `output/`. Run from
`code/business_entity_resolution/src/`:

```bash
pip install polars numpy scikit-learn sparse_dot_topn rapidfuzz unidecode lightgbm
python normalize.py build-dict      # learn the Indic -> English word dictionary
python normalize.py train           # normalize the train sources
python normalize.py test            # normalize the test sources
python blocking.py eval 20000       # held-out recall + labelled candidates for training
python blocking.py test 20          # test candidates (saved per country, resumable)
python lgbm_matcher.py              # cross-validated F0.5 on the held-out records
python lgbm_matcher.py submit       # train, predict, write output/, run the validator
```

Hardware used: 16 GB RAM, 18 CPU threads, no GPU. Test candidate search takes about 2 hours
(France ~4 min, India ~64 min, US ~52 min). Everything else takes minutes.

## Results

| Version | Held-out F0.5 (India / US) | Leaderboard |
|---|---|---|
| **LightGBM + decoy checks** | **0.913 / 0.942** | **0.915** |

This repository reproduces the 0.915 submission exactly: the committed `normalize.py` is the
version that produced the normalized data used for it (verified on 40k re-normalized records).

Candidate recall at top-10 (held-out): India 90.5%, US 95.0%.

## Competition compliance

- Only the provided data is used. There are **no external databases, APIs or geocoding**. The
  Indic dictionary is learned from the training pairs; the static tables are general language
  conventions (street abbreviations, state codes).
- Models and libraries are permissively licensed. LightGBM (MIT) is far below the 8B-parameter
  limit.

## Next steps

- Newer France and global normalization rules (N°/apartment drops, French generic words, street-type
  typo fix, `12bis` split, `5ARL`/`8ordeaux` look-alikes) are kept out of this version and will be
  added with the next submission.
- Run full-recall candidate search. The current run trades 2–4% recall for speed
  (`K_NAME/K_ADDR=20`, `MAX_DF=0.02`).
- Train on ~10× more Source 1 records.
- Add features: address crowding, name rarity, city agreement, house-number distance, the Source
  2/3 record's best vs. second-best owner, synthetic decoys.
- Choose, per Source 1 record, how many candidates to accept, to maximize expected F0.5.
- Replace `unidecode` (GPL) with `anyascii` (ISC) before the final package; add a
  `requirements.txt` and a one-command pipeline script.
