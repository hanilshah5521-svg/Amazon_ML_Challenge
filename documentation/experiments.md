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

## 7. Submission 3 build (26 Sep, leaderboard 0.967)
- **Error analysis of the 0.915 model** (`error_analysis.py`, kept on `main`):
  - Lost points by cause (India / US): candidate-search misses +0.041 / +0.020, false accepts +0.026 / +0.014, false rejects +0.024 / +0.027.
  - 100% of false accepts are decoys that belong to no Source 1 record.
- **Decoy patterns** found by reading the errors:
  - entity-type change (Trust → Private Limited, Partnership → LLP)
  - Private vs public limited
  - compound unit numbers (10-2-10 vs 10-2-12)
  - one business or person word swapped (Brashear → Atherton)
  - leading person name replaced
- **Name ↔ address:** same name with a *conflicting* address means a namesake. Same address with different real-word names means a neighbour. Invented one-word names at the exact address are genuine trade names. Addresses were split into agree / conflict / missing per part.
- **No-match records:** 7% had a strong false candidate. Those decoys are lone single-source candidates with 10× more number conflicts (18.4% vs 1.8%). This led to stage-2 record-level features and the per-record expected-F0.5 decision.
- **Candidate volume:** test has ~20–25% more Source 2/3 records per Source 1 than train, i.e. more decoys. Matches per record, measured by group size, showed no bias against large groups (recall ~87% everywhere).
- **France without labels:**
  - `france_review.py` re-scored French test candidates by category.
  - Inconsistent decisions on "same name + street, different number + legal swap" motivated the group-consistency features.
  - Acronyms (PS, CMP) and business-word swaps (EHPAD → Sportive) became features.
- **Learning curve:** 2k → 16k training records gave +0.003 per doubling, so 200k records were used.
- **Measured ablation** (held-out half B, India / US): base 0.937 / 0.947 → v2 features 0.975 / 0.979 → + stage 2 0.975 / 0.980.
- **Per-state candidate recall at top-10:** India 95.6%, US 97.4%.
- **Leaderboard 0.967.**
- **Dropped this round** (no evidence yet): transitive Source 2↔3 expansion, reverse search, ensembles, embeddings/LLMs, AWS.

## 8. Planned next (for submission 4)
1. Full-recall candidate search (undo the speed shortcut), run overnight.
2. Normalizer v2 (France and global rules), then re-normalize.
3. Train on ~200k Source 1 records instead of 20k.
4. New features: address crowding, name rarity, city agreement, house-number distance, the Source
   2/3 record's best vs second-best owner, synthetic decoys.
5. Choose, per record, how many matches to accept to maximize expected F0.5. Learn a candidate cut
   that keeps F0.5.
6. Replace unidecode with anyascii.

## 9. Improvement round (26 Sep, evening): learned candidate cut, decoy words, more data

Starting point: the 0.967 pipeline, reproduced on a new machine (Python 3.12 instead of 3.10, 6 normalization / 4 feature workers, 15 GB RAM).
Every change below was measured on the **same held-out half B** (200k random train Source 1 records, halves by `default_rng(0)`), kept only if the
weighted score (0.47·India + 0.38·US, normalised) rose by at least 0.0005 and neither country dropped by more than 0.0005. Each experiment ran in its own
work folder so nothing was overwritten. Nothing was tuned on the leaderboard.

### 9.1 Baseline reproduction and loss breakdown
| | India | US |
|---|---|---|
| 24 base features | 0.9364 | 0.9469 |
| + v2 features (stage 1) | 0.9745 | 0.9791 |
| **+ stage 2 (final baseline)** | **0.9751** | **0.9799** |

Where the lost F0.5 sits (baseline final model, half B):

| Cause | India | US |
|---|---|---|
| true match not in the top-15 candidates | 0.0123 | 0.0079 |
| true candidates rejected by the model | 0.0089 | 0.0084 |
| false accepts | 0.0030 | 0.0027 |
| no-match record given a match | 0.0006 | 0.0012 |
| total | 0.0249 | 0.0201 |

Wrong merges are the smallest bucket; candidate recall and rejected true candidates are ~85% of the loss. Baseline test run: 5,731,280 matches, 5.9% of S1 empty,
validator PASS (`output/v1_lgbm_two_stage/`); SHA-256 differs from the reference file (different Python / thread order).

### 9.2 Experiments
| Experiment | India | US | weighted | vs baseline | Verdict |
|---|---|---|---|---|---|
| T3 decoy / noise word features | 0.9762 | 0.9804 | 0.9781 | +0.0011 / +0.0005 | **kept** |
| RK learned candidate cut, K=8 (15 → 8 candidates per S1) | 0.9772 | 0.9813 | 0.9790 | +0.0020 / +0.0014 | **kept** |
| M1 = RK + T3 | 0.9781 | 0.9814 | 0.9796 | +0.0030 / +0.0016 | **kept** |
| M2 = M1 + 200k extra training S1 (400k total) | 0.9790 | 0.9818 | 0.9803 | +0.0039 / +0.0020 | **kept** |
| **M3 = M2 without the inherited `is_india` feature, countries taken from the data (final)** | **0.9788** | **0.9818** | **0.9802** | **+0.0037 / +0.0019** (−0.0002 / −0.0000 vs M2) | **adopted (open-set countries, no measurable cost)** |
| M2 + early-stopped rounds (1749 / 310 rounds instead of 600) | 0.9790 | 0.9820 | 0.9805 | +0.00004 / +0.0002 vs M2 | dropped (< 0.0005, 3× slower) |
| T4 + T5 normalization rules + S1-vocabulary spell correction (on the baseline pipeline) | 0.9736 | 0.9799 | 0.9766 | −0.0015 / +0.0000 | dropped (India worse: more false accepts) |
| M2 with K=10 (10 candidates per S1) | 0.9788 | 0.9822 | 0.9804 | −0.0002 / +0.0003 vs M2 | dropped (no gain, 25% more candidates) |

**T3, decoy / noise words** (`word_lists.py`, features `decoy_added`, `noise_added`, `decoy_dropped`, `n_decoy_added`). A generic word added by the noise is a *decoy* if the
house numbers then disagree (a different business: partners, holdings, group in the US; enterprises, industries, trading, ventures … in India; groupe, holding,
participations, distribution … in France) and *noise* if they agree (center, service, services …). The lists are learned without labels from candidate pairs
(train for India/US, the unlabelled test candidates for France).

**RK, learned cut** (`ranker.py`). Blocking still keeps the top-40 per S1 by `name_sim + addr_sim`. A small LightGBM on 28 cheap pair features (similarities, ranks,
gaps to the best candidate, state relation, house-number overlap, token Jaccards; no per-pair Python) re-ranks them and only the best K go to the matcher, so
`candidate_pairs.tsv` is exactly what the matcher scores. Out-of-fold recall of true matches (200k held-out S1):

| K | India raw cut | India learned | US raw cut | US learned |
|---|---|---|---|---|
| 6 | 0.9215 | 0.9529 | 0.9363 | 0.9620 |
| 8 | 0.9491 | **0.9710** | 0.9657 | **0.9807** |
| 10 | 0.9548 | 0.9742 | 0.9714 | 0.9836 |
| 15 | 0.9601 | 0.9757 | 0.9765 | 0.9853 |
| 40 (ceiling) | 0.9760 | 0.9760 | 0.9854 | 0.9854 |

Eight learned candidates beat fifteen raw candidates in both countries, and the matcher's training set halves (1.6M instead of 3.0M pairs for 200k S1) while
holding more true matches, which is what made the extra training data affordable. The organisers' update says a smaller candidate set per S1 ranks higher; K=8 is
the knee (6 → 8 gains ~1.8 recall points, 8 → 10 only 0.3, 10 → 12 only 0.1).

**M2, more training data.** 200k extra S1 (disjoint from the held-out set, `blocking.py extra`) are added to training and never scored, so results stay comparable.
Loss breakdown, M2 (half B): India rejected 0.0090, not-in-top-K 0.0086, false accepts 0.0028, no-match 0.0005, total 0.0210 (baseline 0.0249);
US rejected 0.0090, not-in-top-K 0.0058, false accepts 0.0022, no-match 0.0011, total 0.0182 (baseline 0.0201).

**Dropped: T4 + T5.** Candidate recall rose (US recall@15 0.9765 → 0.9792, India 0.9601 → 0.9606) but F0.5 did not follow: the extra cleaning made decoys look more
like matches (India false accepts 0.0030 → 0.0045, no-match errors 0.0006 → 0.0017). Which of the two hurts was not isolated.

### 9.3 Error studies (M2 model, out-of-fold)
- **Rejected true pairs** are only 1.8% of true pairs (24,332 of 1,351,943). Typical cases: the candidate has no address (name identical) or the address matches but the name is an
  invented trade name. For empty-address candidates with the same name key the model is essentially perfectly calibrated (mean predicted probability vs true rate:
  1 S1 with that name 0.986 / 0.985; 2 → 0.487 / 0.499; 3 → 0.349 / 0.348; 4+ → 0.043 / 0.043). The 360k candidates with 4+ same-name S1 records (chains) have a 4.3%
  true rate, about 1 true owner in 23 and not identifiable without an address, so most of this loss is an information limit.
- **Decision sensitivity.** Held-out F0.5 is flat around the chosen decision: a threshold 0.05 stricter costs −0.0004 (India) / −0.0001 (US), 0.10 stricter −0.0009 / −0.0005;
  the expected-F rule with miss rate 0 … 0.5 stays within 0.0002. No hedge was applied.
- **Unowned S2/S3 records, train truth vs test predictions** (`analyze_unowned.py`): unowned per S1 1.22 (train) vs 2.56 India / 2.39 US / 2.25 France (test). Orphan clusters
  (S2 + S3 groups sharing name key, first house number and state with no S1 record) are 4.7% (India), 9.5% (US), 16.7% (France) of the unowned test records vs 1.2% / 0.7% in
  train, i.e. 8–27× more per S1 but still a minority; most extra unowned records are lone records. Simulating the shift by dropping S1 records from the held-out set would
  over-create orphan clusters, so a whole-state, decoy-density-matched holdout was not built.

**Open-set countries (M3).** The inherited pipeline used an `is_india` model feature and looped over the hard-coded countries France / India / US when predicting. The task
says country must be an open set, so the final code has no country feature, takes the labelled countries from the training data, iterates over the countries present in the
test files, and gives every country without labels the strictest (highest-threshold) labelled-country decision settings (what France already used). Country-specific *text*
rules (state names, street types, French/Indian address conventions in `normalize.py`) remain, as the task recommends country-specific address patterns. Cost on held-out
data: −0.0002 India, −0.0000 US.

### 9.4 Not done (and why)
- Whole-state holdout with decoy-matched density (T6): design unclear after the unowned-records study, no labels for the test shift.
- Typo-vs-swapped-word alignment, separator-free numbers, no-match classifier, owner competition before the choice (T9): left as the next step; the false-accept + no-match
  buckets are only ~0.004 of the loss.
- Transformer cross-encoder (T10): no time on this machine.
- France-specific normalization (city → region, web-name matching, look-alike legal forms) is only partly covered (T3 word list for France); T4 rules were dropped with T5.

### 9.5 Submissions of this round
| # | Files | Method | Held-out (India / US) | Expected leaderboard | Candidates |
|---|---|---|---|---|---|
| 4 (v1) | `output/v1_lgbm_two_stage/` | reproduction of the 0.967 pipeline | 0.9751 / 0.9799 | 0.967 (reference) | 15 per S1 |
| 4 (v2) | `output/v2_lgbm_two_stage/` | learned cut K=8 + T3 | 0.9781 / 0.9814 | ~0.969–0.970 | 8 per S1 |
| 4 (v3) | `output/v3_lgbm_two_stage/` | learned cut K=8 + T3 + 400k training S1 (still with the `is_india` feature) | 0.9790 / 0.9818 | ~0.970 | 8 per S1 |
| **4 (v4, final)** | `output/v4_lgbm_two_stage/` | same, country-agnostic (no `is_india`, countries from the data) | **0.9788 / 0.9818** | **~0.970 (0.966–0.972)** | 8 per S1 |

## 10. Submission 5 round (27 Sep, branch `patel`): second candidate source before the learned cut

- **Base:** the v4 pipeline (0.973 on the leaderboard), reproduced on a 10-core / 16 GB Mac as **EXP-0**.
  - Held-out half B: India 0.97862, US 0.98188. The documented v4 numbers are 0.9788 / 0.9818.
  - Test matches are within 0.02% of v4 in every country.
- **Leak fix (both experiments):** held-out S1 are cut with out-of-fold ranker scores. v4's cut used a ranker trained on them.
- **EXP-1:** the learned cut's input becomes the v4 per-state TF-IDF top-40 ∪ the `patel` S2/S3→S1 blocker (`code/patel_blocker/`). The ranker gets `our_rank` / `our_rr` features. K stays 8; everything else is unchanged.
- **Recall@8 (out-of-fold):** India 0.9709 → **0.9885**, US 0.9808 → **0.9908**.
- **Half B:**
  - India 0.97862 → **0.98358**; US 0.98188 → **0.98484**.
  - Weighted **+0.00407**, and **+0.00415** on the 50k half-B S1 outside the substitution map's learning half, so there's no leak.
  - The "not in top-K" loss fell from 0.0088 / 0.0060 to 0.0031 / 0.0023.
  - US false accepts rose from 0.0022 to 0.0027.
- **Kept.** Details are in `submissions/05_patel_candidate_union.md`, and the forensic analysis behind the change is in `patel_forensics_and_plan.md`.

## 11. Submission 6 round (27 Sep, branch `patel`): EXP-2, generator-pattern features

- **Change:** EXP-1 plus the 0.976-Hanil matcher features (`features_v2_pattern.py`: 11 generator-pattern features and the FR→US legal-form map; 85 P1 features instead of 74).
  - Candidates, cut and word lists are reused byte-for-byte from EXP-1, so only the feature set changes.
  - `legal_transitions.py` builds its table without the 200k held-out S1 (6.95M pairs).
  - Run: `src/run_exp2.sh`, 32 min.
- **Half B (EXP-1 → EXP-2):**
  - India 0.98358 → **0.98423**; US 0.98484 → **0.98541**; weighted **+0.00061**.
  - Leak-free 50k subset: **+0.00073**.
  - US FP 756 → 624, precision 0.9962 → 0.9969. India FN 5,769 → 5,586, FP 594 → 606.
- **Kept.** It passes all criteria: weighted ≥ +0.0005, no country below −0.0005, and total FP down.
- **Test matches:** France 868,000 (EXP-1 865,515), India 2,701,670 (2,699,835), US 2,247,914 (2,253,736).
- **Submission:** `output/v3_lgbm_two_stage/`. This is the local EXP-2 output, not the 9.5 v3.
  - matching SHA-256 prefix 033b1b3d70ccf7c9, validator PASS (--check-ids).
  - 1,732,544 rows, 98,051 empty.

## 12. Fusion with the 0.9796 file (27 Sep). CORRECTED: the file analysed first was S10 (0.9786); v4/v5 withdrawn, use `output/v6_e2_s9france`
- The 0.9796 file = EXP-1 with 30,364 French pairs removed.
- Removed patterns (French decoys, 5–17× over-represented vs train): name + extra word at a different house number, and acronyms at the same address.
- `output/v4_fusion_e2_h9796fr` = EXP-2 India/US + 0.9796 France.
- `output/v5_fusion_fr_noacronym` = v4 minus the remaining French acronym accepts.
- Analysis: `../../experiments/FUSION/REPORT.md`.

## 13. EXP-3: owner-competition features (27 Sep, 15:12–15:58)
- **Change:** EXP-2 + 6 `oc_*` features from the patel S2/S3→S1 blocker's owner lists over all S1 (`owner_competition.py`, `run_exp3.sh`).
- **Half B:** India 0.98423 → **0.98750**, US 0.98541 → **0.98709**.
  - Leak-free subset: +0.00247.
  - FN −998 / −1,125; FP −188 / −2.
- **Orphan stress** (`stress_exp3.py`; hide h of train S1 from the owner lists):
  - The gain shrinks with h: +0.0022 / +0.0012 at 10%, +0.0015 / +0.0007 at 20%.
  - US turns −0.0003 at 30%.
- **Kept.**
- **Submission:** `output/v8_e3_s9france` = EXP-3 India/US (`output/v7_lgbm_two_stage`) + S9 France.
  - SHA cdb3326c5540f0b7; validator PASS.

## 14. EXP-5: leak fixes + edit / word / number / rare-name features (27 Sep, 16:48–17:52)
- **Leak fixes:**
  - deterministic first house number (hash-order bug in `first_num_logdiff`)
  - legal-transition table built without any of the 400k training S1
  - naive-Bayes and word scores for held-out S1 learned from the other 200k S1 only, 5-fold OOF within those
- **New features** (`exp5_features.py`, 65): 52 edit flags (pattern_miner), `nb_llr`, added/dropped word scores, premise-number / number-edit type / "number changed + same name / legal changed", rarest candidate word, handle overlap.
- **Half B (EXP-3 → EXP-5):**
  - India 0.98750 → **0.98845**, US 0.98709 → **0.98800**; weighted **+0.00093**, leak-free **+0.00101**.
  - FP 418 → 366 / 622 → 480; FN 4,588 → 4,368 / 6,638 → 6,482.
- **Kept.**
- **Submission:** `output/v10_e5_s9france` = EXP-5 India/US (`output/v9_lgbm_two_stage`) + S9 France.
  - SHA 8fc3d0ce211e5f84; validator PASS.
- **Dropped:** the character cross-encoder trained from scratch (loss 0.456 after about 1 fold vs 0.54 for the constant guess; 10 GB).
- **EXP-5 orphan stress** (`15_exp5_orphan_stress.txt`), vs EXP-2 at h = 0 / 10 / 20 / 30%:
  - India +0.0040 / +0.0035 / +0.0029 / +0.0020.
  - US +0.0025 / +0.0021 / +0.0018 / +0.0010.
  - Positive at every level. EXP-3 was −0.0003 in US at 30%.
- **Whole-state holdout** (`16_state_holdout.txt`; 20% of states per country held out, stage 1). EXP-2 → EXP-3 → EXP-5:
  - India 0.98216 → 0.98622 → 0.98765.
  - US 0.98418 → 0.98631 → 0.98756.
- **Recommended final submission:** `output/v10_e5_s9france`.
- **Leaderboard: v10 = 0.983** (27 Sep, ~18:10). Ladder: 0.977 (EXP-1) → 0.9796 (S9) → 0.9786 (S10) → **0.983 (v10)**.

## 15. Team file 0.9822 and v12 (27 Sep, 18:25)
- **The team's 0.9822 file** = the 0.977 file (EXP-1) minus the S9 France removals (identical France) minus **15,397 India / 30,213 US pairs**.
  - The US removals are mostly "same name, different house number" (23,591 with differing first numbers).
  - France is identical to S9, so its +0.0026 over S9 comes only from the India/US removals: test India/US hold number-changed near-duplicates that train does not.
- **EXP-5 already rejects** 21,416 of those 45,610 pairs, and **still accepts 24,194** (India 8,048, US 16,146).
- **v12** = v10 minus those 24,194 pairs (`output/v12_v10_minus_teamIUrule`, SHA 5d9fd557f6cf855d, validator PASS).
- **Package note:** the removal list comes from the team file, so its code or rule must be in the package.
- **EXP-7 started** (18:31): EXP-5 on 800k training S1 (`run_exp7.sh`, `extra2.py`).

## 16. France from EXP-5, EXP-7 (800k S1), final candidates (27 Sep, 18:40–19:42)
- **France from EXP-5** (`code/fusion/build_fr_exp5.py`, teammate's plan).
  - The rule (first house numbers differ AND candidate = S1 name + extra words or an acronym) vs S9 on the EXP-1 file: recall 0.673, precision 0.987.
  - **v13** = v12 India/US + EXP-5 France minus the S9 list (875,505 French pairs).
  - **v14** = v13 + the rule on EXP-5's remaining French pairs (−4,439, all "name + developpement/groupe/participations at a nearby house number"); 871,066 French pairs, SHA 7b8398cf90bcd92e.
- **EXP-7** = EXP-5 on 800k training S1 (`run_exp7.sh`, `extra2.py`).
  - Half B: India 0.98879, US 0.98840. vs EXP-5: +0.00037 weighted, leak-free +0.00029.
  - Below the +0.0005 bar but positive in every cell, with no new feature types.
- **Final candidates**, EXP-7 India/US minus 22,771 teammate-rule pairs:
  - `output/v16_e7_teamIU_e5fr`: France from v14. SHA 282bec37c521c1be.
  - `output/v17_e7_teamIU_s9fr`: S9 France. SHA d414527971d56b00.
  - Choose by v14's LB score.
