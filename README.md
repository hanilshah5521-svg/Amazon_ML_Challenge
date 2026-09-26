# Amazon ML Challenge 2026: Business Entity Resolution (branch `0.967`)

For each business in **Source 1**, find every record in **Sources 2 and 3** that refers to the same
real-world business. Records have a name, an address and a country (US and India in train; France
appears only in test). The metric is macro **F0.5** per Source 1 record.

**This branch contains exactly the code that produced leaderboard score 0.967** (submission 3,
26 Sep 2026). Analysis scripts, research notes and notebooks are on `main`. The history of
everything tried is in [`documentation/`](documentation/).

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
   addresses, top-30 each **within the same state**, plus a pass for records with no state. The
   top 15 per record go to the model.
3. **Features** (`lgbm_matcher.build_features` + `features_v2.py`): string similarities,
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
| `src/run_v2.sh` | One command that runs the whole pipeline |
| `requirements.txt` | Exact library versions (Python 3.10) |

## Reproduce
Put the competition data at `student_resource/dataset/{train,test}/` (it's git-ignored). Then:
```bash
pip install -r code/business_entity_resolution/requirements.txt
bash code/business_entity_resolution/src/run_v2.sh      # ~2.5 h on 16 GB RAM / 18 threads
```
Check the result against the submitted file's SHA-256 prefix, `b5edd37b8508fe00`. Tiny
floating-point or tie-order differences can flip a handful of borderline pairs.

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
