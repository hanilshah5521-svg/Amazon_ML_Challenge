# Amazon ML Challenge 2026: Business Entity Resolution

For each business in **Source 1**, find every record in **Sources 2 and 3** that refers to the same
real-world business. Records have a name, an address and a country (US and India in train; France
appears only in test). The metric is macro **F0.5** per Source 1 record.

**This folder holds the improved pipeline (submission 4, expected ~0.970)**: the 0.967 pipeline (submission 3, kept unchanged in
`code/business_entity_resolution/src_baseline_0967/`) plus a learned candidate cut (8 per Source 1 record), decoy/noise-word features and 400k
training records. The history of everything tried is in [`documentation/`](documentation/).

## Pipeline
```
raw TSVs ─▶ normalize (v2) ─▶ per-state candidate search ─▶ ~70 pair features ─▶ LightGBM stage 1
        ─▶ stage 2 (group consistency) ─▶ per-record expected-F0.5 decision ─▶ one owner per S2/S3 ─▶ TSVs
```

1. **Normalize** (`normalize.py`)
   - Indic scripts go through a dictionary learned from training pairs, with `anyascii` as the
     fallback.
   - Legal forms, alias markers (f/k/a, DBA) and look-alike digits are handled.
   - Address standards: USPS and AFNOR (France) street types, state/region canonicalization,
     Indian city aliases, ordinals, and filler words removed.
2. **Candidate search** (`blocking.py`): character 3-gram TF-IDF on names and word TF-IDF on
   addresses, top-30 each **within the same state**, plus a pass for records with no state; the top 40 per record are kept.
   **Learned cut** (`ranker.py`): a small LightGBM on cheap pair features re-ranks the 40 and only the best **8** go to the matcher.
   `candidate_pairs.tsv` is exactly this set.
3. **Features** (`lgbm_matcher.build_features` + `features_v2.py` + decoy/noise words from `word_lists.py`): string similarities,
   entity-type changes, compound house numbers, rarity-weighted distinctive vs generic words,
   invented trade name, acronym, website name, state agree/conflict/missing, name and address
   crowding, and same-name group features.
4. **Model** (`model_v2.py`): LightGBM stage 1 (pairs). Stage 2 re-scores each pair using the
   stage-1 results of the record's other candidates.
5. **Decision:** per record, accept the top-n candidates that maximize expected F0.5 (n can be 0).
   Each Source 2/3 record goes only to its best Source 1 record. France uses the stricter US
   settings.
6. **Output:** `output/v<N>_lgbm_two_stage/` (never overwritten), checked with the organizers'
   validator using `--check-ids`.

## Files
| File | Role |
|---|---|
| `src/common.py` | Paths and loaders |
| `src/normalize.py` | Normalization rules and the Indic dictionary |
| `src/blocking.py` | Per-state candidate search (`eval` for training records, `test` for test) |
| `src/pair_features.py` | Legal-form and house-number relations |
| `src/lgbm_matcher.py` | The 24 base pair features (used by v2) |
| `src/features_v2.py` | The v2 feature groups |
| `src/model_v2.py` | Two-stage training, ablation report, decision rules, test prediction |
| `src/evaluation.py` | F0.5 scorer, decision helpers, TSV writer, versioned output folders |
| `src/ranker.py` | Learned candidate cut (fit, cut train/test) |
| `src/word_lists.py` | Decoy / noise word lists per country (no labels) |
| `src/pick_k.py` | Picks the cut size K from the ranker's recall table |
| `src/run_final.sh` | One command that runs the whole final pipeline |
| `requirements.txt` | Exact library versions (Python 3.10) |

## Reproduce
Put the competition data at `student_resource/dataset/{train,test}/` (it's git-ignored). Then:
```bash
pip install -r code/business_entity_resolution/requirements.txt
bash code/business_entity_resolution/src/run_final.sh   # ~2 h 15 min on 16 GB RAM / 24 threads
```
Lower `NORM_WORKERS` / `FEAT_WORKERS` (defaults 6 / 4) if RAM is tight. Tested with Python 3.12 (the file that scored 0.967 came from Python 3.10, SHA-256 prefix
`b5edd37b8508fe00`; a different Python and thread order change tie-breaking, see `documentation/experiments.md` section 9). The final file's (`output/v4_lgbm_two_stage/`) SHA-256 prefix is
`00ccf1b7586a3332`.

## Results (held-out half B of 200k training Source 1 records, macro F0.5)
| Pipeline | India | US | Candidates per S1 |
|---|---|---|---|
| 0.967 pipeline (reproduced) | 0.9751 | 0.9799 | 15 |
| + learned cut (K=8) | 0.9772 | 0.9813 | 8 |
| + decoy/noise words | 0.9781 | 0.9814 | 8 |
| + 400k training records | 0.9790 | 0.9818 | 8 |
| **country-agnostic (no `is_india`), final** | **0.9788** | **0.9818** | 8 |

Expected leaderboard about 0.970 (0.969-0.971); France has no labels.

## Results
| | India | US | Leaderboard |
|---|---|---|---|
| Held-out F0.5 (half B of 200k training records) | 0.975 | 0.980 | **0.967** |
| Candidate recall at top-10 | 95.6% | 97.4% | |

## Compliance
- Only the provided data is used: no external databases, APIs or geocoding.
- The Indic dictionary is learned from the training pairs. Static tables are general language
  conventions (USPS/AFNOR abbreviations, state codes).
- All libraries are permissively licensed (MIT/BSD/ISC). LightGBM is far below the
  8B-parameter limit.
