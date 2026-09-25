# Experiments log

This is everything we tried, in order. "Kept" means it's in the current best pipeline. "Dropped"
means it was tested and rejected, with the reason.

## 1. Data understanding (25 Sep)
- **Train:** Source 1 has 2.2M records (US 1.32M, India 0.88M); Sources 2 and 3 have 5.0M and 5.3M.
  **Test:** Source 1 has 1.73M (India 47%, US 38%, **France 15%**, and France appears only in test).
- Only **5.6%** of Source 1 records have no match. They average **3.46 matches** (max 11).
- **Every Source 2/3 record belongs to at most one Source 1 record** → the "one owner" rule.
- A match never crosses countries → search each country separately.
- Source 2/3 names are 11–19% non-Latin. Indian names appear in 9 scripts (10 languages); the
  script follows the business's state.
- **39% of Source 1 records share their name with another business in the same country**, so the
  address has to decide.
- The pattern explorer (`explore_neighbors.py`, top-10 by address, full pool) found top-10 recall
  of India 85.4% and US 78.1%. Misses came from empty or vague addresses, US hamlet vs town names,
  noisy house numbers, and trade/website names.

## 2. Normalization (`normalize.py`)
Measured on ~207k true pairs, with each step kept:

| Step | Result |
|---|---|
| Raw text | 4.7% of names identical |
| Lowercase + punctuation → space (not deleted) + transliteration + abbreviations | 29.7% identical |
| Learned Indic → English dictionary (positional alignment of true pairs; 100% equal token counts) | Indic names 96.9% identical in core |
| Mined fixes (`edge_miner.py`): generic words (`center` is noise-added 72% of the time), alias split, TG ≡ AP (the #1 substitution), Indian city aliases, ordinals, USPS suffixes, noise address words, Indian premise words | name core identical **64.7%**, address overlap 0.841 |

- **User-flagged edge cases, checked on data:**
  - **Roman numerals:** converted in addresses only. In names they stay, because of look-alike
    noise ("XI" vs "Xl").
  - **"(India)" in names:** dropped from the core.
  - **"1 800 I Care 4 U":** digits are kept, and a lone "I" is not converted.
- **Bugs found by testing:**
  - The state map replaced text anywhere, which broke "Kansas City". Fixed to match whole address
    components only.
  - Deleting punctuation instead of replacing it with a space broke Indic vowel signs.
- **Transliteration fallbacks tried** for unseen words:
  - unidecode, ITRANS, HK and IAST all managed only ~0.8 Jaro-Winkler.
  - A **consonant-skeleton key** raised exact matches from 3.8% to 64%. Not yet used in the
    pipeline.
- **France (unsupervised, `france_miner.py`, `france_patterns.py`):**
  - "N°" appears in 11.7% of Source 2/3 records and ~0% of Source 1.
  - Street types have typos (aveue, avnue).
  - "12bis" is glued.
  - French generic words appear (participations 2.4% vs 0%).
  - One of our own mapping rules was inconsistent (all/allee).
  - House numbers agree in 79% of pseudo-pairs.
  - 12% of French Source 1 records sit at crowded addresses.
  - **These rules are written but not yet in the submitted pipeline** (`work/normalize_v2.py`).

## 3. Candidate search (`blocking.py`)
- A fixed similarity threshold does **not scale**: 4.5 → 227 → ~1,400 candidates per record as
  the pool grows. **Top-K per record** fixes this.
- Ranking by name + address sum beat either signal alone: US top-10 recall 97.5% vs 89.4%
  (address) and 75.6% (name).
- **Full-pool evaluation, slow version** (top-50 per search): India recall@10 94.2%, US 97.1%;
  recall at K=100 is 95.7% / 98.2%. The curve bends at K ≈ 8.
- **Speed:** the slow version took 37 minutes for France alone, and India/US would have taken hours.
  **Fast version** (top-20 per search, dropping tokens in >2% of the pool): France 4 min, India
  64 min, US 52 min, but recall@10 fell to **India 90.5%, US 95.0%**. Accepted for the first
  submissions; to be undone overnight.
- **Why matches are missed** (at K=10): mostly *not found by either search* or *match has no
  address*, then different trade names.

## 4. Deciding matches
| Experiment | Result | Decision |
|---|---|---|
| Name + address sum, one threshold | India 0.802, US 0.864 (slow candidates) | baseline |
| Stricter "gate" for records with no match | +0.001–0.002 only | dropped |
| **Discovery:** 99% of high-scoring false candidates for no-match records are **planted decoys** (same name and street, different legal form or unit number) | legal conflict: 0.3% of true pairs vs 17% of false; all numbers shared: 85% vs 12% | → features |
| + number check + legal-form check | India 0.869, US 0.896 | kept (as features) |
| Legal form as a bonus as well as a penalty | hurt F0.5 badly (agreeing decoys got boosted) | fixed: penalty only |
| Per-record cut (keep candidates within 0.5 of the best), heuristic matcher | same F0.5 with ~3.4 candidates per record | used in submission 1 |
| **LightGBM** (24 features, grouped 5-fold) | **India 0.913, US 0.942** | **submission 2** |
| Per-record cut **before** LightGBM | India 0.886, US 0.919 (−2.5 points) | dropped; top-10 used |

## 5. Research (`research_notes/`)
Five parallel literature reviews: candidate search, matchers and clustering, normalization,
competitions and benchmarks, and industry systems. Main takeaways:
- Character n-gram TF-IDF top-K is at or near the state of the art for candidate search.
- The top Foursquare 2022 teams used a GBDT with relative features plus a graph post-processing
  step.
- LLM and deep matchers aren't worth it on CPU.
- Licence notes: avoid GPL/AGPL (unidecode, abydos, Aksharamukha, Zingg) and libpostal (it
  bundles OpenStreetMap data).

## 6. Operational notes
- 16 GB RAM is the main constraint. Heavy steps run one country at a time, and runs save each
  country as it finishes.
- A run that ran out of memory and runs stopped when the session ended were restarted. No
  results were lost.
- The held-out F0.5 estimate matched the leaderboard for submission 2 (0.915).

## 7. Planned next (for submission 3)
1. Full-recall candidate search (undo the speed shortcut), run overnight.
2. Normalizer v2 (France and global rules), then re-normalize.
3. Train on ~200k Source 1 records instead of 20k.
4. New features: address crowding, name rarity, city agreement, house-number distance, the Source
   2/3 record's best vs second-best owner, synthetic decoys.
5. Choose, per record, how many matches to accept to maximize expected F0.5. Learn a candidate cut
   that keeps F0.5.
6. Replace unidecode with anyascii.
