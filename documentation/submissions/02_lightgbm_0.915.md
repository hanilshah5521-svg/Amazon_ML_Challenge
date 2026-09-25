# Submission 2: LightGBM matcher (leaderboard 0.915)

- **Submitted:** 25 Sep 2026, ~23:55 IST
- **Leaderboard (public):** **0.915** (current best)
- **Output file fingerprint:** SHA-256 of `output/matching_results.tsv` starts with `8b1b22a6b36b15ea`
- **Reproduce:** repository commit `148c1a1` (or later with the same `src/`); follow the root README.
  Library versions are in `code/business_entity_resolution/requirements.txt`.

## Pipeline
1. **Normalization** (`normalize.py`, the version pinned in the repository)
   - Indian-script names go through a word dictionary learned from the training pairs (1,347
     words; covers 96% of Indic tokens in test).
   - Legal forms are canonicalized, alias markers split (the text after f/k/a / DBA is kept as the
     name), generic words dropped for `name_key`, look-alike 1/0 digits fixed, and Roman numerals
     converted in addresses.
   - Addresses: USPS and French abbreviations, state/region canonicalization (TG ≡ AP, Nord ≡
     Hauts-de-France), Indian old/new city names, ordinals up to 99, and noise words
     (unit/apt/pmb/cdp…) removed.
2. **Candidate search** (`blocking.py`), per country
   - top-20 by character 3-gram TF-IDF on `name_core` (spaces removed), plus top-20 by word TF-IDF
     on the address
   - tokens present in >2% of the pool are dropped
   - merged and ranked by name + address similarity; **top-10 per record** go to the matcher
     (10 candidates per record, no per-record cut)
3. **Features** (24): name/address similarities, house-number relation (all shared / one side
   missing / conflict), legal-form relation (same family / missing / conflict), RapidFuzz ratios
   on core/key/no-space names, address token-set ratio, empty-address flag, length difference, and
   features relative to the record's other candidates (gap to best, score/best, number of close
   candidates, name/address rank).
4. **Model:** LightGBM (binary; 400 rounds, learning rate 0.05, 63 leaves). Trained on ~200k
   labelled candidate pairs from the 20k held-out train records.
5. **Decision:** accept if p ≥ threshold (India 0.65, US 0.70, **France 0.70**, the stricter of
   the two). Each Source 2/3 record goes only to its highest-probability Source 1 record.
6. **Output:** 1,732,544 rows; 6.7% of records empty; ~3.1 matches per matched record. Official
   validator: PASS.

## Measured results
| | Held-out F0.5 (5-fold grouped, out-of-fold) |
|---|---|
| India | 0.913 |
| US | 0.942 |
| France | no labels; ~0.85 implied by the leaderboard |

- The held-out estimate predicted the leaderboard well (0.915), so local evaluation can be trusted
  for India and the US.
- Most important features: blocking score, house-number relation, score relative to the best
  candidate, address token-set ratio, legal-form relation.

## Known limitations (targets for the next submission)
- The fast candidate search loses 2–4% of true matches (top-10 recall: India 90.5%, US 95.0%).
- The model was trained on only 20k Source 1 records.
- Newer France and global normalization rules are not included (kept in `work/normalize_v2.py`).
- 10 candidates per record. A fixed per-record cut cost 2.5 points, so shrinking the candidate set
  must be learned instead.
- `unidecode` is GPL-licensed. Replace it with `anyascii` before the final package.
