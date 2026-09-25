# Submission 1: hand-weighted score (leaderboard 0.725)

- **Submitted:** 25 Sep 2026, ~23:40 IST
- **Leaderboard (public):** **0.725**
- **Status:** superseded by submission 2. Its code (`baseline.py`) was removed from the repository.

## What it did
1. The same normalization and candidate search as submission 2: fast search, top-20 by name
   and top-20 by address, merged, top-10 kept.
2. **Per-record cut:** kept only candidates whose score was within 0.5 of that record's best
   score (~4.0 candidates per record).
3. **Score** = name similarity (character 3-gram TF-IDF cosine) + address similarity (word
   TF-IDF cosine). There were no house-number or legal-form checks.
4. **Decision:** accept if score ≥ 1.6, then give each Source 2/3 record only to its highest-scoring
   Source 1 record.
5. Official validator: PASS. 6.2% of records were left empty.

## Why it scored low
- **My estimate (0.80–0.84) was wrong.** It came from the rule measured on the older, slower
  candidate search. On the candidates actually used, held-out F0.5 was **India 0.757, US 0.800**.
- **France was probably worse still** (~0.4–0.5, inferred from the leaderboard). France has crowded
  addresses (e.g. 101 businesses at one "Maison des associations"), generic names repeated across
  cities, and planted decoys. A plain name + address sum accepts all of these.
- **It had no protection against decoys:** same name and street, but a different legal form or unit
  number.

## Lessons
- Estimate scores with the **exact** candidates and rule being submitted.
- Decoy checks (legal form, house numbers) and a learned matcher are essential. See submission 2.
