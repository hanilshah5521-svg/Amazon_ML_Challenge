# Pairwise matchers and post-matching clustering/assignment for entity resolution (AMLC 2026)

Context: S1 (clean reference) -> 0..many matches in S2/S3; each S2/S3 record belongs to at most one S1 record; macro F0.5 per S1; current hand-weighted scorer ~0.87-0.90 held-out, candidate oracle ~0.97-0.98, leaderboard top ~0.989. CPU only, 16 GB RAM, ~48h, MIT/Apache-2.0, <=8B params.

## Q1. GBDT pairwise matchers: features that matter, reported results

### Takeaway
In the closest public analogue (Kaggle Foursquare Location Matching 2022: noisy POI name+address+geo, many-to-many matches), every top-10 solution used a candidate-generation stage followed by a GBDT (LightGBM/XGBoost) on pairwise string-similarity + embedding + rank features, often ensembled with a BERT cross-encoder and followed by graph post-processing. A learned GBDT over ~30-60 similarity features is the standard, CPU-friendly step up from a hand-weighted score.

### Cited Findings
- 7th place Foursquare (1083 teams): 3 stages - retrieval (32 candidates/query from 5 generators: haversine kNN, weighted haversine+embedding, word-level BoW, char-level BoW, name-embedding cosine), LightGBM prediction, graph post-processing. — [Future Architect blog, 7th place](https://future-architect.github.io/articles/20220720a/)
- 7th place LightGBM features: Gestalt (difflib) ratio, Levenshtein, Jaro-Winkler on processed name/address; ROUGE-N / ROUGE-L; categorical agreement (country, state, city, category); haversine distance; embedding cosine and k-means cluster ids/distances. Model: 4096 leaves, lr 0.1, 2000 iterations, with sample weighting "emphasizing avoiding false positives". — [Future Architect blog](https://future-architect.github.io/articles/20220720a/)
- 7th place per-stage max-IoU: retrieval alone 0.9778, retrieval + post-processing 0.9935; custom embeddings raised candidate max-IoU from 0.7582 (USE baseline) to 0.8997 at precision@32. — [Future Architect blog](https://future-architect.github.io/articles/20220720a/)
- Team re:waiwai (1st place): candidate generation -> LightGBM with few features (filter) -> LightGBM with engineered features (Levenshtein, Jaro-Winkler, SVD embeddings) -> BERT models with iterative fine-tuning. Uehara's 4-stage pipeline (text+geo candidates, LightGBM filter, two transformer cross-encoders, GNN post-processing) improved from 0.907 to 0.946. Team 2:30 ensembled LightGBM + XGBoost + BERT. — [Foursquare blog "Finding the right POI match"](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- A different public solution (hobbitlab): FAISS cosine over concatenated multilingual embeddings (XLM-R, LaBSE, paraphrase-multilingual-mpnet) gives 50 candidates + BallTree geo candidates; XGBoost features include TF-IDF cosine for name/address/category, normalized Levenshtein, geo distances, and rank/"hit_count" features across candidate ranks. Progression: 0.922 baseline -> +0.006 model fusion -> +0.003 XGBoost features -> +0.009 NaN imputation -> 0.940+. — [hobbitlab GitHub README](https://github.com/hobbitlab/Foursquare-Location-Matching/blob/main/readme.md)
- 16th place Foursquare: ~50 features; most important were cosine similarities between text fields, distance rank (e.g. 2nd-closest vs 150th-closest neighbor) and Jaccard similarity between names. — [Kaggle #16 solution notes (via search snippet)](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/silogram-a-few-notes-on-16-solution)
- Splink (Fellegi-Sunter) applies term-frequency adjustments: a match on a rare value is stronger evidence than a match on a common value; the TF adjustment is inversely proportional to value frequency and added as an independent term per comparison. — [Splink docs: term frequency](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html)

### Inferences
- For our data the feature set to build (all CPU-cheap with rapidfuzz/sklearn): name char-3/4gram TF-IDF cosine, word TF-IDF cosine, rapidfuzz ratio/partial_ratio/token_sort/token_set, Jaro-Winkler, prefix match length, IDF-weighted token overlap (sum of IDF of shared tokens / sum of IDF of union) - this is the Splink TF-adjustment idea applied to tokens; same features on "core name" (legal suffix stripped) vs full name; legal-form equal/conflict/missing as categorical; address TF-IDF cosine + token_set; house/unit-number exact agreement, conflict, one-side-missing (separately for each numeric token type); postcode/city/state equality; country; string lengths and length ratio; "name is acronym of other"; website-style normalized equality; script flag (Latin/Devanagari/other) and transliterated-name similarity.
- Train with sample weights or a threshold that favours precision (as 7th place did) since the metric is F0.5.
- Expected gain: moving from a hand-weighted linear score to GBDT on the same signals typically closes a large part of the gap to the candidate oracle; in Foursquare, feature/model work added single-digit points on top of strong retrieval. A realistic target from GBDT + relative features + assignment post-processing is ~0.93-0.96 given an oracle of 0.97-0.98 (inference, not measured).

### Gaps
- Could not retrieve the full 1st-place Foursquare writeup text (Kaggle page did not render), so its exact feature list and per-stage gains are unknown.
- No source found giving a systematic feature-importance ranking specifically for company-name matching (Magellan paper not fetched).

## Q2. Relative / listwise features and learning-to-rank

### Takeaway
Rank-of-candidate features (e.g. "this candidate is the 2nd nearest / 2nd most similar for this query") were among the top features in Foursquare solutions; no ER paper found that isolates their gain with an ablation, but practitioner evidence is consistent.

### Cited Findings
- Distance rank (2nd closest vs 150th closest) was one of the most important features in the 16th-place model. — [Kaggle #16 solution notes](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/silogram-a-few-notes-on-16-solution)
- hobbitlab XGBoost used rank-derived features: hit_count_02..05/sum with min/max aggregations over candidate ranks, and embedding-distance differences between rank positions (d0_d1 ... d0_d5, i.e. gap between the best and k-th candidate). — [hobbitlab README](https://github.com/hobbitlab/Foursquare-Location-Matching/blob/main/readme.md)
- 7th place used learning-to-rank for one candidate generator (weighted haversine + embedding cosine ranked via "ranking learning"). — [Future Architect blog](https://future-architect.github.io/articles/20220720a/)

### Inferences
- Add, per pair (s1, t): rank of t among s1's candidates by each key similarity; gap to best similarity for s1; score / max score for s1; number of s1's candidates above 0.8 name-sim; and the reverse direction: rank of s1 among all S1 candidates of t, gap between t's best and second-best S1 (critical because each t belongs to <=1 S1). The reverse-direction features directly target the decoys (two S1 records competing for the same S2 record).
- Two-pass stacking works well on CPU: train GBDT pass 1, compute per-query/per-target aggregates of pass-1 probabilities (max, 2nd max, rank, count>0.5), train pass 2. Out-of-fold predictions are required to avoid leakage.
- LambdaMART (LightGBM `objective=lambdarank`, group = S1 id) is a plausible alternative but since we need an absolute accept/reject decision (0..11 matches, 5.6% none), binary logloss + relative features is simpler; LTR scores are not calibrated for thresholding.

### Gaps
- Found no peer-reviewed ablation quantifying rank/relative features for ER or LambdaMART vs binary classification for ER.

## Q3. Deep and LLM matchers: accuracy, licenses, CPU cost

### Takeaway
Fine-tuned cross-encoders (Ditto-style) remain the strongest supervised pairwise matchers; generative LLMs <=8B are weaker than fine-tuned small PLMs zero-shot and are far too slow for CPU over hundreds of thousands of pairs. Ditto code is Apache-2.0; Jellyfish is CC BY-NC (disallowed); Llama is not MIT/Apache. For a 48h CPU budget, a small cross-encoder is at best a re-ranker on the uncertain band, not the main matcher.

### Cited Findings
- Ditto casts EM as sequence-pair classification with fine-tuned BERT/DistilBERT/RoBERTa; up to +29% F1 over previous SOTA on benchmarks; F1 96.5% on matching two company datasets of 789K and 412K records. Optimizations: domain-knowledge injection (highlighting spans), summarization of long strings, data augmentation with difficult examples. — [Ditto, VLDB 2021 / arXiv 2004.00584](https://arxiv.org/abs/2004.00584)
- Ditto repository code license: Apache License 2.0 (the VLDB paper PDF itself is CC BY-NC-ND, which applies to the paper, not the code). — [megagonlabs/ditto LICENSE](https://github.com/megagonlabs/ditto/blob/master/LICENSE); [VLDB PDF](https://www.vldb.org/pvldb/vol14/p50-li.pdf)
- Peeters et al. (Entity Matching using LLMs): zero-shot F1, open LLMs vs fine-tuned PLMs - WDC Products: Llama-3.1-70B 83.67, Mixtral-8x7B 53.37, RoBERTa 77.53, Ditto 84.90; Walmart-Amazon: 84.85 / 70.44 / 87.02 / 86.39; Amazon-Google: 73.99 / 40.98 / 79.27 / 80.07; Abt-Buy: 89.84 / 82.20 / 91.21 / 91.31. Llama-3.1-70B runtime ~0.54 s/prompt zero-shot (GPU). Conclusion: open LLMs can match hosted LLMs given a little task data or rules. — [Peeters, Steiner, Bizer, arXiv 2310.11244](https://arxiv.org/html/2310.11244v4)
- Fine-tuning LLMs for EM: Llama-3.1-8B on WDC Products zero-shot 53.36 -> fine-tuned 69.19; GPT-4o-mini 81.61 -> 83.41; Llama-8B transfer to Abt-Buy 87.34, Walmart-Amazon 65.65, Amazon-Google 50.00. LoRA r=64, alpha 16, lr 2e-4, 10 epochs on up to 8 RTX6000 GPUs. — [Steiner, Peeters, Bizer, arXiv 2409.08185](https://arxiv.org/html/2409.08185v1)
- Survey of 8 open instruction-tuned LLMs for EM included Llama-3.1-8B, Qwen2-7B, Mistral-7B, Mixtral-8x7B, Command-R-35B, Flan-T5-XXL, Flan-UL2, Solar-10.7B. — [Match, Compare, or Select? (COLING 2025)](https://arxiv.org/html/2405.16884v3.pdf)
- AnyMatch: fine-tuned GPT-2 (124M) zero-shot on unseen datasets, mean F1 81.96 over 9 datasets vs MatchGPT/GPT-4 86.36, Jellyfish 77.83, Ditto (zero-shot transfer) 66.05; 3,899x lower inference cost; throughput 694K tokens/s vs Jellyfish-13B 26.7K tokens/s (4x A100). — [AnyMatch, arXiv 2409.04073](https://arxiv.org/html/2409.04073v1)
- Jellyfish-7B = Mistral-7B-Instruct-v0.2 fine-tune, license CC BY-NC-4.0 (non-commercial); Amazon-Google F1 81.69 vs GPT-4 74.21; Walmart-Amazon 84.91 vs 90.27. 8B variant is Llama-3-8B based. — [HF NECOUDBFM/Jellyfish-7B](https://huggingface.co/NECOUDBFM/Jellyfish-7B); [Jellyfish paper](https://arxiv.org/abs/2312.01678)
- Qwen2.5 licenses: 0.5B, 1.5B, 7B, 14B, 32B Apache-2.0; 3B is Qwen Research License; 72B Qwen License. — [Qwen2.5 blog](https://qwenlm.github.io/blog/qwen2.5/)

### License/feasibility summary (for our rules)
| Option | License | <=8B | CPU feasible in 48h? |
|---|---|---|---|
| LightGBM/XGBoost + rapidfuzz features | MIT / Apache-2.0 / MIT | yes | yes, minutes |
| Ditto code (megagonlabs) | Apache-2.0 (code) | yes | fine-tune on CPU slow; DistilBERT/MiniLM re-ranker on few-10k pairs feasible |
| AnyMatch (GPT-2 124M) | GPT-2 weights MIT (not verified here) | yes | inference OK, training marginal |
| Jellyfish 7B/8B/13B | CC BY-NC-4.0 | 7/8B yes, 13B no | VIOLATES license rule |
| Llama-3/3.1-8B | Llama community license | yes | VIOLATES MIT/Apache rule |
| Qwen2.5-0.5B/1.5B/7B | Apache-2.0 | yes | 0.5-1.5B only for small re-rank sets |
| Qwen2.5-3B | Qwen Research License | yes | VIOLATES |
| Mixtral-8x7B | (Apache-2.0, not verified here) | NO (47B) | VIOLATES size |

### Inferences
- Our fields are short (name + address), so a 22M-66M param cross-encoder (MiniLM/DistilBERT class) would run roughly hundreds of pairs/second on CPU; fine-tuning 1 epoch on ~50-100K pairs is likely several hours on a laptop - feasible but risky in 48h. Best use: score only pairs in the GBDT uncertain band (e.g. p in 0.2-0.8) and feed its logit back as a GBDT feature.
- Generative 7B LLMs on CPU (llama.cpp, 4-bit) run at a few tokens/s; with hundreds of thousands of candidate pairs this is infeasible except for a tiny hard subset, and published F1 for 8B models is below fine-tuned PLMs anyway. Not worth it for the time budget.
- Decoys (Pvt Ltd vs LLP, unit number) are exactly what explicit features catch better than a generic PLM; Ditto's "domain knowledge injection" (tagging numbers / legal forms) exists because vanilla PLMs miss them.

### Gaps
- Could not verify license text of roberta-base / distilbert / MiniLM / GPT-2 / Mistral-7B weights in this session (generally reported MIT/Apache, needs checking on HF model cards).
- Did not find published CPU throughput numbers for Ditto; estimate above is an inference.
- DeepMatcher, HierMatcher, Unicorn numbers not collected (older/weaker than Ditto or GPU-oriented).

## Q4. Post-matching: clustering, transitivity, one-to-many assignment

### Takeaway
Graph post-processing gave measurable gains in Foursquare (7th place: 0.9778 -> 0.9935 max-IoU including post-processing; another team used a GNN for 0.907 -> 0.946 overall). For our structure (clean S1, each S2/S3 owned by <=1 S1) the right post-processing is a "best-owner" (max-both style) assignment rather than Hungarian 1:1 or plain connected components, plus using S2<->S3 links to propagate.

### Cited Findings
- 7th place post-processing: Union-Find over pairs above threshold, remove high-betweenness edges (NetworkX) as structural noise, then add pairs within distance 2 in the resulting graph (limited transitive closure). — [Future Architect blog](https://future-architect.github.io/articles/20220720a/)
- Uehara's pipeline used GNN post-processing as the 4th stage (overall 0.907 -> 0.946). — [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- Hassanzadeh et al. (Stringer, VLDB 2009) compared Partitioning (connected components), Center, Merge-Center, Star, Ricochet, Cut, Articulation Point, Markov Clustering (MCL), Correlation Clustering: MCL was consistently top; single-pass Center and Merge-Center gave comparable or better results than Cut and Correlation Clustering; Ricochet good only with uniform duplicate distribution. — [Framework for Evaluating Clustering Algorithms in Duplicate Detection](https://dl.acm.org/doi/10.14778/1687627.1687771) (details via search summary; PDF text could not be parsed)
- Multi-source ER (Saeedi, Rahm): clustering on the similarity graph can transitively infer links and eliminate unlikely ones; "max-both" assignment assigns a new entity to the highest-similarity cluster (above a threshold) only if no other entity from the same source has higher similarity to that cluster; MSCD scheme handles mixes of clean (duplicate-free) and dirty sources. — [Comparative Evaluation of Distributed Clustering Schemes for Multi-source ER](https://link.springer.com/chapter/10.1007/978-3-319-66917-5_19); [Incremental Multi-source ER for KG completion](https://pmc.ncbi.nlm.nih.gov/articles/PMC7250616/)
- TransClean: uses transitive consistency of pairwise decisions across multiple sources to iteratively remove false positives; average +24.42 F1 over pairwise matchers (DistilBERT, CLER) in multi-source settings. — [TransClean, arXiv 2506.04006](https://arxiv.org/abs/2506.04006)

### Inferences (design for our constraint)
- Hungarian 1:1 is wrong here: an S1 has 0..11 matches. The constraint is only "each t in S2/S3 has <=1 owner", so: for each t, keep only its argmax-probability S1 (if p >= threshold, optionally also require margin over 2nd-best S1). This is the max-both idea restricted to one side and is O(pairs). It directly kills decoys where two S1 records (Pvt Ltd vs LLP, unit 101 vs 102) compete for the same t.
- Plain connected components / full transitive closure is dangerous with decoys: one wrong S2-S3 edge merges two S1 entities. Use S1-anchored propagation instead: if t2 in S2 is confidently matched to s1 and t3 in S3 has a high S2-S3 similarity to t2 but a weak direct score to s1, raise p(s1,t3) (e.g. feature = max over already-accepted siblings of sim(t3, sibling)); still enforce t3's single owner. Add as a pass-2 GBDT feature ("max similarity to accepted matches of this S1") rather than a hard rule; this is the "leverage S2<->S3 links" requirement.
- Consistency check (TransClean-like): if t is accepted for s1 but has low similarity to all other accepted matches of s1 and to s1 itself is borderline, drop it - precision-weighted metric rewards this.

### Gaps
- No quantitative source for gains of best-owner assignment specifically; must be validated on our held-out split.

## Q5. Threshold selection for F0.5 under per-query macro averaging; no-match queries

### Takeaway
With calibrated probabilities, the F-optimal threshold is tied to the achievable F (for F1, threshold = F*/2); for F0.5 the precision emphasis pushes it higher. But with macro-averaged per-S1 F0.5 and a "predict nothing" rule for no-match queries, the threshold should be tuned directly on the exact metric on held-out data, ideally with per-query decisions.

### Cited Findings
- Lipton, Elkan, Narayanaswamy: for calibrated probabilities, the F1-optimal threshold equals half the optimal achievable F1; they derive the general relation between best F1 and threshold and discuss multilabel (macro vs micro) settings. — [Thresholding Classifiers to Maximize F1 Score, arXiv 1402.1892](https://arxiv.org/abs/1402.1892)

### Inferences
- Generalizing the same argument to F-beta gives threshold = F*_beta / (1 + beta^2) for a per-item expected-utility decision, i.e. F*/1.25 for beta=0.5 (so with F* ~0.95, threshold ~0.76) - derived, not cited; verify empirically.
- Macro per-S1 means a query-level decision is better than a global threshold: for each S1, sort candidates by p and pick the prefix size k (including k=0) maximizing expected F0.5 under the model's probabilities (the "expected F-measure maximization" approach, computable exactly for <=~15 candidates by enumerating k). k=0 yields expected score = P(no true match) = prod(1-p_i), which handles the 5.6% singletons naturally.
- Needs well-calibrated probabilities: use out-of-fold GBDT probabilities with isotonic calibration, and tune a global offset on held-out macro F0.5 afterward.

### Gaps
- No ER-specific paper found on macro-per-query F-beta optimization; the expected-F prefix method is standard in multilabel literature but not cited here.

## Q6. Hard-negative mining, training-set construction, decoys

### Takeaway
Train the matcher on exactly the blocking output (all candidates per S1, labelled from ground truth), which naturally contains hard negatives; augment/upweight decoy-like negatives. Top systems explicitly selected difficult pairs and weighted to avoid false positives.

### Cited Findings
- Ditto augments training data with difficult examples and injects domain knowledge by highlighting key spans (e.g. numbers). — [Ditto arXiv](https://arxiv.org/abs/2004.00584)
- AnyMatch selects difficult pairs via an AutoML filter, adds attribute-level examples, and controls label imbalance. — [AnyMatch](https://arxiv.org/html/2409.04073v1)
- Foursquare 7th place trained LightGBM on the 32 retrieved candidates per query with sample weights emphasizing avoidance of false positives; 1st place (re:waiwai) used a cheap LightGBM filter stage before the expensive feature model. — [Future Architect blog](https://future-architect.github.io/articles/20220720a/); [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- LLM fine-tuning for EM used 2,000-11,364 negatives vs 500-8,471 positives per dataset. — [arXiv 2409.08185](https://arxiv.org/html/2409.08185v1)

### Inferences
- Construct training pairs = all blocking candidates for train S1 records (positives from GT, rest negatives); group K-fold by S1 id (and by country, keep India/US mix; France is test-only, so hold out cross-country checks and avoid country-identity features that would not transfer).
- Synthesize decoy negatives from positives: swap legal form (Pvt Ltd <-> LLP / Inc <-> LLC), change unit/house number, and add them as negatives with higher weight; mirror augmentations as positives (abbreviate, drop suffix, reorder words, drop address parts) to teach invariances.
- Two-stage like re:waiwai: cheap filter (current hand score or small LightGBM) to cut candidates, then full feature model.
- Monitor precision on decoy-type negatives as a separate validation slice.

### Gaps
- No source quantifies the gain of synthetic decoy negatives specifically.
