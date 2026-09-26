# Harsh V (leaderboard 0.973): forensic analysis, and the plan to beat it

- **Sources:** the `harshv` branch of `hanilshah5521-svg/Amazon_ML_Challenge` (commit `e02dbb1`, cloned read-only to `reference/harshv/`). I read every pipeline module: `normalize`, `blocking`, `ranker`, `lgbm_matcher`, `pair_features`, `features_v2`, `word_lists`, `model_v2`, `evaluation`, `common` and `run_final.sh`, plus its documentation. Our own measurements are in `experiments/DATA-FORENSICS/01–11`.
- **Labels:** [READ] = seen in the code · [DOC] = the author's documented measurement · [MEAS] = measured by us here · [INF] = inference.

## 1. What the 0.973 pipeline actually does [READ]
```
raw TSV ─► normalize (per-country rules + Indic dictionary) ─► per-STATE blocking, S1 → S2∪S3 pool (name char-3gram TF-IDF top-30
∪ address word TF-IDF top-30, + no-state pass top-10), keep top-40 by name_sim+addr_sim ─► learned cut: a small LightGBM ranker keeps
the best K=8 per S1 (= candidate_pairs.tsv) ─► ~70 pair features ─► LightGBM stage 1 (600 rounds, 5-fold GroupKFold by S1) ─► stage 2
(group features from the other candidates' stage-1 p) ─► per-record expected-F0.5 top-n choice (n may be 0) ─► one owner per S2/S3
(highest score) ─► TSVs
```

### Normalisation (`normalize.py`)
**Indic script:**
- A word dictionary is learned from train true pairs (positional alignment, equal token counts), with `anyascii` as the fallback.
- 17 native-script state names are mapped to English.

**Names:**
- Alias splitting (f/k/a, dba, "doing business as", …) and the `M/s` prefix are removed.
- Look-alike digits are fixed (1→l, 0→o, 5/8/6→s/b/g at word start).
- Legal forms are canonicalised through `LEGAL`, which covers US/India forms and the French sarl/sas/sasu/sa/eurl/sci/snc/ste/ets/ei.
- Stopwords are dropped.
- There are three views:
  - `name` (full)
  - `name_core` (own name, no legal forms or stopwords, consecutive duplicates removed)
  - `name_key` (core minus about 60 GENERIC words, including French participations/holding/groupe/fils/freres)

**Addresses:**
- The state is canonicalised **only when a whole comma component is the state**, so "Kansas City" is safe.
- France: departments map to regions (nord / pas-de-calais → hdf, gironde → naq, loire-atlantique → pdl).
- About 120 static USPS and AFNOR street-type abbreviations, plus number words, ordinals, glued `12bis`, `94 B`→bis, and one-edit street-type typos.
- India: city aliases (Calcutta→Kolkata, …) and premise words (plot/flat/shop → no).
- Noise tokens are dropped (unit/apt/pmb/box/county/…), plus French no/appt/etage/bat/residence.

### Blocking (`blocking.py`)
- The direction is **S1 → pool**, per country **and per state**.
- Name: `TfidfVectorizer(char 3-gram)` on `name_core` with spaces removed. Address: word TF-IDF. `sparse_dot_topn` finds top-30 each, plus top-10 against pool records with no state.
- The union is ranked by `name_sim + addr_sim` and the **top 40** are kept. `via` ∈ {name, address, both}.

### Learned cut (`ranker.py`)
- A LightGBM (300 rounds, 63 leaves) on 28 cheap vectorised features: similarities, ranks, gaps to the best, `state_rel`, `key_equal`, `core_equal`, shared numbers, first-number equality, token Jaccards and the counts `n_key_equal` / `n_first_num_eq`.
- It keeps the **top K=8 per S1**.
- [DOC] Recall@8, learned: India 0.971, US 0.981. The raw @8 cut gave 0.949 / 0.966, and the ceiling @40 is 0.976 / 0.985.

### Pair features, about 70 (`lgbm_matcher` + `features_v2` + `pair_features`)
- **24 base features:**
  - blocking similarities, rank, `via`
  - `legal_rel` (legal family: same 1 / missing 0.5 / conflict 0)
  - `num_rel` (number-set containment)
  - rapidfuzz ratio / token_sort / token_set / partial on core, key ratio and equality, no-space ratio
  - address token_set, address empty, length difference
  - gap and ratio to the S1's best, n_close, name and address rank, best_key
- **Entity type:** `ent_added`, `ent_dropped`, `ent_changed`, `private_diff`, `legal_equal`.
- **Compound house numbers:** "10-2-10" is kept whole. Features: exact, subset, shared, counts, *one part changed*, truncation, within 3, range contains, log first-number distance.
- **Words:** first distinctive word equal, *invented trade name*, acronym, website form, web-contains, place word in the name. IDF-weighted **shared vs unmatched** name and address tokens (sum and max), and generic unmatched counts.
- **Decoy / noise words** (`word_lists.py`, *label-free*): among pairs with the same name key and the same street where exactly one word is added, a word whose house numbers then **disagree** under 30% of the time is a *decoy* word (partners/holdings/group…), and one that agrees over 70% of the time is *noise*. The India/US lists come from train candidates; the **France list comes from unlabelled test candidates**.
- **Other:**
  - `state_rel` (agree / conflict / missing)
  - crowding: how many S1 / pool records share the name key, and how many share the address
  - group features: same-key candidates of this S1, best address among namesakes, address gap to best
  - `is_s2`

### Model (`model_v2.py`)
- **Stage 1:** LightGBM with lr 0.05, 127 leaves, `min_child_samples` 50, feature fraction 0.7, bagging 0.8, λ2 1, **600 rounds**, 5-fold GroupKFold by S1.
- **Stage 2:** the same model plus 15 group features from the other candidates' stage-1 p: rank, max, second-best, gap, the counts above 0.5 and above 0.8, `src_strong` (S2 *and* S3 among the strong candidates), sum, **the top candidate's** cnum_subset / address-empty / ent_changed, and **deviation from the other strong candidates' majority** on legal-equal, number subset and key-equal.
- Training uses 400k S1: 200k held-out (val) plus 200k extra. Stage 2 is adopted only if it wins in both countries.

### Decision (`decide_expected_f`)
- Candidates are sorted per S1. E[F(n)] ≈ 1.25·Σtop-n p / (0.25·T + n), with T = Σp·(1+miss).
- It accepts the best n if that beats P(no match) = Π(1−p), which makes it an explicit **singleton decision**.
- The rule is expected-F versus a plain threshold, and `miss` ∈ {0, 0.05, 0.1, 0.2} is chosen **on half A** and reported on **half B**.
- Countries without labels (France) get the **strictest** settings.
- **One owner per S2/S3**: the highest score wins.

## 2. Why it works — ranked contributors (author ablations [DOC] + our measurement [MEAS])

| # | Component | Evidence | Errors it prevents |
|---|---|---|---|
| 1 | **v2 decoy/structure features** (legal family, entity type, compound numbers, IDF words, crowding) | Stage 1: 0.936 → 0.975 India, 0.947 → 0.979 US (+3.5 pts) [DOC] | **Planted decoys**: same name and street, different legal form or unit number. 100% of the 0.915 model's false accepts were decoys [DOC]. |
| 2 | **Explicit legal/number logic transfers to test and France** | Our 0.98 → **0.963** vs their 0.98 → 0.973 (see §4) [MEAS] | Decoys are about 2× denser in test (unowned per S1: train 1.22, test ≈ 2.3) [DOC + MEAS] |
| 3 | Learned cut K=8 | Recall @8 up 2.2 / 1.5 pts with 47% fewer candidates; +0.0020 / +0.0014 held-out [DOC] | Fewer decoys reach the matcher, and more true matches do |
| 4 | Expected-F0.5 top-n and singleton decision | Chosen over the threshold rule on half A [READ] | Singletons: accepting nothing is scored explicitly |
| 5 | Per-state blocking | True pairs always share the state (0 exceptions in 300k) [DOC] | Namesakes in other states never become candidates |
| 6 | Country rules (AFNOR, regions, French legal forms, French decoy list, strict France) | – | French decoys (sa vs sarl) and department vs region mismatches |
| 7 | Stage-2 group consistency | +0.0006 / +0.0008 [DOC] | A candidate that deviates from its S1's other strong candidates (number, legal, key) |

Interaction [INF]: the learned cut and the decoy features work together. The cut keeps the matcher's input to 8 hard candidates, and the decoy features are what separate a true match from a same-street decoy inside that set.

## 3. Weaknesses of the 0.973 pipeline
**Loss on held-out half B, M2 model [DOC]:**

| Cause | India | US |
|---|---:|---:|
| True candidate rejected | 0.0090 | 0.0090 |
| **True match not in top-K** | **0.0086** | **0.0058** |
| False accepts | 0.0028 | 0.0022 |
| No-match S1 given a match | 0.0005 | 0.0011 |
| **Total** | 0.0210 | 0.0182 |

- **Candidate recall @8** is 0.971 India / 0.981 US [DOC]. **Our** deep-rerank blocking gets **0.980 / 0.984** at about 8.5 per S1 [MEAS] (same data, but query direction S2/S3 → S1).
- **Rejected true matches** are mostly records with no address and a chain name (4+ same-name S1s: 4.3% true rate, an information limit) and **invented trade names at the right address** [DOC]. For the second group, same-source **sibling address similarity** separates true from false with AUC 0.877, against 0.768 for the S1 address [MEAS, `02_…txt`]. Harsh V has no sibling features.
- **Validation leakage** [READ]:
  - (a) The Indic dictionary is built from *all* train pairs, including half B.
  - (b) The ranker's final model is trained on *all* 200k val S1, including half B, and then used to cut half B. The half-B candidate cut is in-sample, so held-out recall and F0.5 are optimistic.
  - This is consistent with their own 0.98 → 0.973 gap.
- **The validation doesn't reflect test decoy density.** They measured about 2× unowned records per S1 in test but didn't simulate it [DOC §9.3].

## 4. Why our EXP-06b (validation 0.981) scored only 0.963 [MEAS]
- **Ruled out:**
  - Orphan distractors (0.1% merged in the 5.7-per-S1 simulation).
  - US/India score-distribution shift: near-zero-scored records look like train distractors.
  - Overfitting: the strict held-out check matched.
- **Found** (`11_decoy_signals_in_our_predictions.txt`):
  - Accepted test pairs with a **legal-form conflict *and* a house-number conflict** are **0.38% in the US and 0.30% in France**, against **0.02% in train**, where their precision is 62.5%. That's 15–19× more.
  - US number-conflict accepts rise from 6.8% to 10.0%.
  - Reading them: textbook decoys ("cultuelle loisirs **sa**, **19** av du vallon" vs "cultuelle loisirs **sarl**, **10** av du vallon") accepted at p = 0.78–0.96.
- **Why our pipeline is exposed:**
  - (i) It has **no legal-form or entity-type features**; French forms are unseen tokens.
  - (ii) Its number features are weak (first-number equality and shared count, with no compound numbers).
  - (iii) **Every S2/S3 record looks for an owner**, so each extra test decoy gets a chance to attach. Harsh V instead fills 8 slots per S1 and chooses with expected-F.
  - (iv) Our stage-2 **sibling address features likely *boost* same-street decoys** [INF].
- **Conclusion:** keep Harsh V as the base (rule 11). Our additions must be decoy-safe.

## 5. Component comparison
| Component | Harsh V 0.973 | Our EXP-06b (0.963) | Better | Decision |
|---|---|---|---|---|
| Normalisation | Rich static and learned rules, country-specific (USPS/AFNOR/India), 3 name views | Transliteration + NFKD + learned S2/S3→S1 token map (learned on half of S1) | Harsh V (France, legal forms) | **KEEP Harsh V** |
| Transliteration | Learned Indic dictionary + anyascii | Brahmic offset table + learned substitutions | Similar; ours avoids the leak | KEEP Harsh V; optionally add our map (later) |
| Blocking | S1 → pool, per state, TF-IDF name/address top-30, top-40 | S2/S3 → S1, IDF key index, deep re-rank for low-confidence records | **Ours on recall** (0.980 / 0.984 vs 0.971 / 0.981 at ~8) | **COMBINE: union** |
| Candidate cut | Learned ranker K=8 | Relative pruning (8.5 per S1) | Harsh V (per-S1 budget, decoy-safe) | KEEP the cut, feed it the union |
| Features | ~70 decoy-oriented | ~40 similarity + 22 context | **Harsh V** | KEEP Harsh V |
| Sibling / family context | None (group features only within an S1's list) | Same-source sibling address/name similarity, family support | Ours on recall, but decoy-unsafe | **MODIFY → decoy-aware** (later) |
| Model | 2-stage LightGBM, 600 rounds | 2-stage LightGBM, ≤2000 rounds | Equivalent (their ES test: +0.0002) | KEEP Harsh V |
| Decision | Expected-F top-n + singleton; strict for unseen countries | Per-record threshold with family thresholds | Harsh V (singletons, per-S1) | **KEEP Harsh V** |
| Exclusivity | One owner, highest score | Built in (query direction) | Same | KEEP |
| Validation | Half A / B of 200k S1 (leaky cut) | 2-fold S1 OOF + strict check | Ours on rigour | Fix the cut leak |

## 6–9. Error, recall, false-positive, false-negative and singleton findings (numbers above)
- **Candidate recall:** their @8 recall is 0.971 / 0.981 [DOC], against ours at 0.980 / 0.984 [MEAS]. Our remaining misses are 52% no-address chain names, which is the information limit.
- **False positives:** decoys, i.e. same name and street with a different legal form or number. They're 2× denser in test and 15–19× more frequent in our accepted pairs than in train [MEAS].
- **False negatives:** rejected true candidates (0.009, largest bucket), then not-in-top-K (0.0086 / 0.0058).
- **Singletons:** 0.0005 / 0.0011 [DOC]. That's small, and the expected-F rule already handles it.

## 10–12. Keep / remove / add
- **KEEP (from 0.973):**
  - all normalisation
  - per-state blocking
  - the learned cut, K=8
  - every v2 feature and the decoy word lists
  - 2-stage LightGBM
  - expected-F decision
  - strict settings for unseen countries
  - one owner
- **REMOVE (from ours):**
  - our pipeline as the base
  - family-aware thresholds (they suit our per-record rule, not an S1-level decision)
  - using our sibling features *as they are*
- **ADD, in order of evidence:**
  - (1) **Candidate union**: our deep-rerank candidates added to their top-40, before the learned cut.
  - (2) Remove the leak in the cut (out-of-fold cut for held-out S1).
  - (3) Decoy-aware sibling features in stage 2.
  - (4) Test-prior calibration of p for decoy density (EM prior estimate, label-free).

## 13. Final proposed architecture
```
DATA ─► NORMALIZATION [KEEP] ─► BLOCKING: Harsh V per-state TF-IDF top-40 [KEEP] ∪ our deep-rerank candidates flipped to S1→cand [NEW]
─► LEARNED CUT K=8 on the union (ranker retrained; `via` gains a third source; OOF cut for held-out S1) [MODIFY]
─► ~70 FEATURES [KEEP] ─► STAGE 1 [KEEP] ─► STAGE 2 + decoy-aware family features [NEW, EXP-2]
─► EXPECTED-F0.5 top-n + singleton decision, strict for unseen countries [KEEP] (+ test-prior calibration [NEW, EXP-3])
─► ONE OWNER [KEEP] ─► SUBMISSION (8 candidates per S1)
```

## 14. Ablation sequence
All experiments use the same held-out half B, the same seeds, and their `model_v2.evaluate` (half A picks the decision rule, half B reports).

| Exp | Change | Hypothesis | Precision / recall | Keep if |
|---|---|---|---|---|
| **EXP-0** | Reproduce Harsh V v4 here; also report the cut with an OOF ranker | Baseline on this machine; leak size | – | Reference |
| **EXP-1** | + candidate union, K=8 unchanged | Cuts the "not in top-K" loss (0.0086 / 0.0058) | R ↑, P ≈ | Weighted ≥ +0.0005 and no country worse than −0.0005 (their rule) |
| EXP-2 | + decoy-aware family features in stage 2: family support; same-source sibling address similarity **with number and legal agreement against the family consensus** | Rescues rejected trade-name matches without boosting decoys | R ↑; must not raise the false-accept bucket | Same rule, **and** false accepts not up |
| EXP-3 | + EM-estimated test prior used to rescale p before the decision | Test decoy density is 2× train | P ↑ on test | Only if it is neutral on half B (it is a test-time correction) |
| EXP-4 | + our learned S2/S3→S1 token map as an extra blocking view | More India recall | R ↑ | Same rule |

## 15. Expected bottleneck after these changes [INF]
- Records with no address and a chain name (information limit).
- The unknowable France accuracy.

## 16. Files to change (in a **copy** of the harshv code, `work_hv/`, never the original)
- `blocking.py`: new `union` mode that reads `work_e04/{train,test}/cands.parquet` and maps rids → entity ids (S1 → cand), adding `our_rr` / `our_rank` columns and `via` bit 4.
- `ranker.py`: add `our_rr`, `our_rank` and `via` to `FEATS`; `fit` gains an OOF cut for val S1.
- `model_v2.py`: no change for EXP-1.
- `run_final.sh`: step 1b (union).
- New: `union_candidates.py`, about 60 lines.

## 17. NEXT RUN — EXP-1 (with EXP-0 in the same pass)
- **Changes:** Harsh V v4 pipeline unchanged, except that the ranker's input candidate set = Harsh V top-40 ∪ our EXP-04 candidates (flipped to S1 → cand, for the 200k val + 200k extra train S1 and all test S1). The learned cut stays at **K=8**.
- **Model:** Harsh V stage 1 + stage 2 as is (600 rounds, same parameters, seeds 7/0).
- **Features:** Harsh V's ~70, plus `our_rr` / `our_rank` / `via` bit 4 in the ranker only.
- **Blocking:** per-state TF-IDF top-40 ∪ our IDF deep re-rank (low-confidence records: top-200 re-ranked, keep 8; confident: keep 3; rel 0.8).
- **Threshold strategy:** unchanged. Expected-F vs threshold is chosen on half A; unseen countries use the strictest.
- **Why it should improve:** "not in top-K" is their largest *fixable* loss (0.0086 India / 0.0058 US), and our candidates reach 0.980 / 0.984 recall at a similar budget. K stays 8, so decoy exposure and candidate-file size don't grow.
- **Validation:** half B of the same 200k val S1 (`default_rng(0)` halves). Both EXP-0 and EXP-1 are also reported with an OOF ranker cut, to remove the in-sample cut leak. Report macro F0.5, precision, recall, FP / FN, singleton accuracy and recall@8 by country and source.
- **Success:** the weighted half-B F0.5 (0.47·India + 0.38·US) rises by ≥ 0.0005 over EXP-0, neither country drops by more than 0.0005, and the false-accept bucket doesn't grow.
- **Rollback:** anything else. Harsh V v4 (0.973) stays the submission.
- **Cost:** EXP-0 about 3–3.5 h on this 10-core / 16 GB machine (their 2 h 15 min was on 24 threads), plus EXP-1 about 1.5 h (the ranker and matcher rerun; normalisation and blocking are shared).
