# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [YOUR TEAM NAME]  
**Team Members:** [LIST ALL TEAM MEMBERS]  
**Submission Date:** 26–27 September 2026

---

## 1. Executive Summary
A two-stage LightGBM matcher scores candidate pairs produced by per-state TF-IDF search followed by a learned cut that keeps only the 8 best candidates per Source 1
record. Features include string similarity, house-number and legal-form relations, name rarity, "decoy vs noise word" indicators learned without labels, and group-level
stage-2 features; a per-record expected-F0.5 rule decides how many matches to accept (including none) and each Source 2/3 record is given to a single Source 1 owner.
Held-out macro F0.5 is 0.9788 (India) and 0.9818 (US) with 8 candidates per record, against 0.9751 / 0.9799 with 15 candidates for the previous version of the pipeline.

---

## 2. Methodology

### 2.1 Problem Analysis
- Train has 2.2M Source 1 (US 1.32M, India 0.88M), 5.0M Source 2, 5.3M Source 3 records; test has 1.73M Source 1 (India 47%, US 38%, France 15%, France only in test).
- Only 5.6% of Source 1 records have no match; they average 3.46 matches (max 11). Every Source 2/3 record belongs to at most one Source 1 record ("one owner"); matches never cross
  countries and share the state whenever both sides have one.
- Noise: typos in Source 2/3 (Source 1 is almost typo-free), legal-form changes and drops, alias markers (f/k/a, dba, née), website/handle names, invented trade names,
  generic words added by the noise (a *noise* word such as "center" leaves the business unchanged; a *decoy* word such as "holdings" marks a different business),
  Indic-script names in nine scripts, truncated or reordered addresses, empty addresses (~4%).
- 39% of Source 1 names are shared with another business in the same country, so the address decides; planted decoys (same name and street, different number or legal form)
  are the main source of false merges. Test has about twice the unowned Source 2/3 records per Source 1 record as train.

### 2.2 Solution Strategy
**Approach Type:** Blocking (per-state TF-IDF) + learned candidate cut + two-stage gradient-boosted classifier + expected-F0.5 decision.  
**Core Innovation:** (1) a cheap learned cut that keeps 8 candidates per record with higher recall than the raw top-15 (India 0.971 vs 0.960, US 0.981 vs 0.977), which also
halves the matcher's training and inference cost; (2) per-country decoy/noise word lists discovered without labels from house-number agreement of candidate pairs;
(3) a per-record expected-F0.5 decision that can predict nothing, so records with no match are protected.

---

## 3. Candidate Generation (Blocking)

- **Normalization:** lower-case, accent stripping (anyascii, ISC), punctuation to spaces, Indic-script words mapped to English with a dictionary learned from the training pairs
  (1,347 words) plus anyascii as fallback, legal-form and stop-word removal (`name_core`, `name_key`), USPS/AFNOR street types, state canonicalization, ordinals, look-alike digits.
- **Blocking keys used:** per country and within the same state: character 3-gram TF-IDF on `name_core` (spaces removed) and word TF-IDF on the normalized address, top-30 each,
  plus a pass of the top-10 against pool records with no detectable state (France has regions on every Source 1 record but not on a third of Source 2/3 records).
  The union is ranked by `name_sim + addr_sim` and the **top 40** are stored.
- **Learned cut:** a small LightGBM (`ranker.py`, 28 cheap features: similarities, ranks, gaps to the best candidate, state relation, house-number overlap, token Jaccards; no
  per-pair string-distance loops) re-ranks the 40 and only the **best 8 per Source 1 record** are kept. Out-of-fold recall of true matches on 200k held-out records:
  K=8 India 0.9710 / US 0.9807 (raw cut 0.9491 / 0.9657), K=10 0.9742 / 0.9836, ceiling at 40: 0.9760 / 0.9854.
- **Candidate pairs generated (test):** 1,732,544 × 8 = 13,860,352 (France 2,075,616, India 6,479,888, US 5,304,848); before the cut the per-state union held 20.1M / 53.8M / 43.4M pairs.
  `candidate_pairs.tsv` is exactly the set the matcher scores.
- **How we ensured true matches were not lost:** no common-word dropping in the TF-IDF search (full recall per state); the cut is learned on labelled pairs and evaluated
  out-of-fold; the choice K=8 is the knee of the recall curve (6 → 8 gains 1.8 points, 8 → 10 only 0.3 and no F0.5).

---

## 4. Matching Model

**Features used:**
- Name features: Jaro-Winkler, token sort/set, partial ratios, char n-gram TF-IDF cosine, first-word equality, acronym, invented trade name, website name, IDF-weighted shared and
  unmatched distinctive words, generic-word categories, legal/entity type added / dropped / changed, **decoy-word added / noise-word added / decoy dropped** (per-country lists
  learned without labels from house-number agreement of candidate pairs).
- Address features: house numbers kept whole (exact, subset, one part changed, truncation, within 3, range), IDF-weighted street/locality words, state agree / conflict / missing,
  empty-address flags.
- Other: name and address crowding (how many Source 1 / pool records share the name key or address), same-name candidates of the record and which has the best address.
- **Stage 2 (group) features:** stage-1 probabilities of the record's other candidates (strong-candidate counts, both sources present, second best, gap, warning flags of the top
  candidate, deviation from the group majority).

**Model type:** LightGBM (MIT licence), trained from scratch on the provided data only, no pretrained or external models: stage 1 on ~75 features and stage 2 on 15 more,
600 rounds, learning rate 0.05, 127 leaves, 5-fold cross-validation grouped by Source 1 record, trained on 400k Source 1 records (3.2M candidate pairs).  
**Threshold selection method:** per country and per record: either a probability threshold or the expected-F0.5 rule (accept the top-n candidates that maximize
1.25·TP / (0.25·T + n), n may be 0), picked on half A of the held-out records and reported on half B; then each Source 2/3 record is given to its highest-scoring Source 1 record.
The model has no country feature; countries are taken from the data and any country without labels (France) uses the strictest labelled-country decision settings.
Country-specific handling exists only in the text normalization (state names, street types, French and Indian address conventions).

---

## 5. Results & Error Analysis

- **F0.5 Score (macro), held-out half B:** India **0.9788**, US **0.9818** (weighted 0.9802). Steps: previous pipeline 0.9751 / 0.9799 → learned cut 0.9772 / 0.9813 →
  + decoy/noise words 0.9781 / 0.9814 → + 400k training records 0.9790 / 0.9818 → country-agnostic (no country feature) 0.9788 / 0.9818. Test output: 5,736,957 matches,
  5.8% of Source 1 records left empty (training truth 5.6%).
- **Common false positives (wrong merges):** planted decoys: same name and street with a different house number or legal form, and generic-word swaps ("partners", "holdings",
  "group"). False accepts + no-match errors cost 0.004 (India) and 0.003 (US).
- **Common false negatives (missed matches):** (a) true match not in the 8 candidates (0.0086 India, 0.0058 US); (b) candidates with no address whose name is shared by many Source 1
  records — the model is calibrated there (a chain name with 4+ Source 1 records has a 4.3% true rate), so most of this loss is an information limit; (c) invented trade names at the exact address.
- **Not done / tested and dropped:** early-stopped boosting rounds (no gain), extra normalization rules plus S1-vocabulary spell correction (India worse), K=10 candidates (no gain).

---

## 6. Conclusion
Keeping fewer but better-chosen candidates (a learned cut) made the matcher both cheaper and more accurate, and decoy/noise word features plus more training data added the rest.
The remaining errors are dominated by information limits (candidates without addresses whose names are shared by many businesses) and by France, which cannot be measured locally.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`: `normalize.py` (text normalization, Indic dictionary), `blocking.py` (per-state candidate search), `ranker.py` (learned cut),
`word_lists.py` (decoy/noise words), `lgbm_matcher.py` + `features_v2.py` + `pair_features.py` (features), `model_v2.py` (two-stage model, decision, prediction, validator call),
`evaluation.py` (F0.5, output writer). Entry point: `bash src/run_final.sh` (about 2 h 15 min on 16 GB RAM); it writes `output/v<N>_lgbm_two_stage/matching_results.tsv` and
`candidate_pairs.tsv`. Dependencies are pinned in `requirements.txt`.

### B. Additional Results
Full experiment log with every kept and dropped change, recall tables and error studies: `documentation/experiments.md` (section 9) in the repository.

---

**Note:** No external databases, APIs, geocoding or internet data were used. Statistics computed from the unlabelled test files (IDF, vocabulary, France decoy-word list) are unsupervised.
