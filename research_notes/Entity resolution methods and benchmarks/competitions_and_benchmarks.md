# Entity-matching competitions and benchmarks: winning pipelines and achievable scores

Context: AMLC 2026 business ER (S1 query -> matches in S2/S3; names + addresses + country; macro F0.5 per query). Leader 0.989, our baseline ~0.86-0.88, oracle ceiling on our candidates ~0.97-0.98. CPU-only, ~48h.

Note on access: Kaggle write-up pages (kaggle.com/.../writeups, /discussion) render empty to the fetch tool, so the Foursquare/Shopee details below come from secondary sources: the Foursquare official blog, the 7th-place team's blog, GitHub READMEs, and a Japanese Shopee recap. Every number is cited to where it was found.

## Q1. Kaggle "Foursquare - Location Matching" (2022): top solutions

### Takeaway
Every top solution used the same design: (1) cheap candidate generation from spatial kNN plus name kNN (TF-IDF, char n-gram or embedding), about 20-50 candidates per record, reaching a max IoU of about 0.977-0.99; (2) a LightGBM stage with a few cheap features to prune candidates; (3) a second GBDT with rich string-similarity features, and/or a fine-tuned multilingual transformer (XLM-R) cross-/bi-encoder; (4) graph post-processing (union-find/transitivity with edge pruning, or a GNN). Top final IoU was about 0.95. Post-processing added large gains: one team's candidate ceiling went from 0.978 to 0.9935 after graph merging, and a GNN post-processor took the 2nd-place-listed solution from 0.907 to 0.946.

### Cited Findings
- Task: match about 1M+ POI entries (name, street address, coordinates, category) to the same real-world POI. Metric: mean IoU (Jaccard) between predicted and true match sets per entry — [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/); [Kaggle competition](https://www.kaggle.com/competitions/foursquare-location-matching)
- Training set had >1M entries. Test set about 600k — [hobbitlab README](https://github.com/hobbitlab/Foursquare-Location-Matching/blob/main/readme.md)
- **Placement conflict:** The Foursquare blog lists Philipp Singer 1st (deep learning only: ArcFace metric-learning embeddings for candidates, then a bi-encoder NLP model trained on those candidates, stages blended), Yuki Uehara 2nd, Team re:waiwai 3rd and Team 2:30 4th — [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/). But the Kaggle write-up slug is "re-waiwai-1st-place-solution" — [Kaggle](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution) — and a search summary calls Team 2:30's approach the winning one. The final private LB may have been re-scored after the leakage issue below; I could not resolve this.
- **Uehara (listed 2nd):** four stages. (1) Candidates from text similarity plus geographic proximity. (2) LightGBM filtering. (3) Pairwise match classification with two transformer LMs. (4) GNN node classification as post-processing. **Score went from 0.907 to 0.946** — [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- **re:waiwai:** Stage 1 builds candidates from text similarity plus geographic proximity, then a memory-efficient LightGBM on limited features prunes them. Stage 2 is LightGBM with expanded features (Levenshtein, Jaro-Winkler, embedding SVD). Stages 3-4 are BERT-type (XLM-RoBERTa) models with iterative fine-tuning, plus post-processing — [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/). A search-snippet summary of a top solution says the **top 40 candidates per ID** from the first LightGBM go to the second LightGBM, with name-embedding SVD features — [search summary referencing TheoViel/kaggle_foursquare & top write-ups](https://github.com/TheoViel/kaggle_foursquare)
- **Team 2:30:** geographic proximity plus TF-IDF candidates, then a **Transformer-based blocking stage** that cuts candidates while keeping recall high, then an ensemble of LightGBM, XGBoost and BERT models — [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- **4th place (Theo Viel et al., per repo):** two-level GBDT ensemble (CatBoost, LightGBM, XGBoost): level 1 prefilters, level 2 predicts. Has a separate post-processing module. The submitted model used fewer candidates than the shared pipeline for memory reasons — [TheoViel/kaggle_foursquare](https://github.com/TheoViel/kaggle_foursquare)
- **7th place (Future Architect blog), the most detailed numbers found:**
  - Retrieval: 4 haversine neighbours + 12 from a regression-weighted mix of haversine and embedding cosine + 4 word-level name matches + 8 char-level name matches + 4 name-embedding cosine = **32 candidates per query**.
  - **Max IoU was 0.9778 from retrieval alone and 0.9935 with retrieval plus post-processing.**
  - Features: Levenshtein, Jaro-Winkler, ROUGE-N/L, category/geo agreement, "Word Tour" 1-D embedding-cluster distance, k-means++ cluster features.
  - Model: LightGBM with 2^12 leaves, lr 0.1, 2000-3000 iterations, custom sample weights that penalise false positives, 5-fold CV.
  - Post-processing: UnionFind graph, edges removed by betweenness centrality, pairs within graph distance 2 kept.
  - About 67% of test overlapped train (leakage), so they re-evaluated on a deduplicated set.
  — [Future Architect blog (7th/1083)](https://future-architect.github.io/articles/20220720a/)
- **Another top solution (via search snippet):** max IoU 0.97675 with 25 candidates per id. Candidates came from BallTree: 15 lat/lon neighbours, 15 name+lat/lon neighbours, 10 lat/lon+category neighbours. Candidates were thinned with per-rank-band percentiles (e.g. keep the top 15% of pairs among neighbours 1-10 but only the top 0.05% among neighbours 190-199). Extra candidates came from joins on rounded lat/lon, and on city plus name. Another solution reached max IoU >0.99 using concatenated paraphrase-xlm-r-multilingual-v1 and msmarco-bert-base-dot-v5 embeddings — [search summary of Zhihu recap / #16 / #21 write-ups](https://zhuanlan.zhihu.com/p/555976163), [#16 notes](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/silogram-a-few-notes-on-16-solution), [#21](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/bulian-ai-21st-place-solution)
- **Mid-table embedding solution (hobbitlab), showing the size of each step:** 50 FAISS cosine candidates plus BallTree spatial candidates; 4 fine-tuned BERTs (XLM-R-large, LaBSE, paraphrase-multilingual-mpnet, RemBERT), each 320-dim, concatenated; XGBoost with 35+ features (distances, similarities, edit distances, hit counts). Scores: **0.922** (mBERT baseline), **0.928** (+XLM-R fusion), **0.931** (+XGBoost features), **0.940** (null-value filling + 4-BERT fusion) — [hobbitlab README](https://github.com/hobbitlab/Foursquare-Location-Matching/blob/main/readme.md)

### Inferences
- The pattern that matches our setup: blend several candidate sources (geo/country block + word TF-IDF + char n-gram + embeddings), aim for about 25-40 candidates at about 0.98 ceiling, then GBDT, then graph post-processing. The 7th-place 0.978 -> 0.9935 jump shows that transitive/graph expansion also lifts the candidate ceiling. For us: if S2 record A matches S1 query q, and S3 record B strongly matches A, B is probably a match for q even if B was not in q's candidate list.
- Foursquare POI data was real and noisy, and top scores were about 0.95 IoU. A leader at 0.989 on our task points to cleaner or synthetic noise (see Q4). The gap between our ~0.87 and our ~0.975 ceiling is mostly a matcher and thresholding problem, not a candidate-generation problem.
- Sample weighting against false positives (7th place) fits F0.5, which rewards precision.

### Gaps
- Exact final private-LB IoU for places 1-4 and per-stage ablations for re:waiwai / Team 2:30: Kaggle write-up bodies were not retrievable.
- Which team was actually 1st on the final private LB: sources conflict (see above).

## Q2. Other competitions: Shopee (2021), SIGMOD contests (2020-2022)

### Takeaway
Shopee winners relied on metric-learned embeddings plus heavy neighbourhood post-processing: mutual-rank matching, rank-gap rules, query expansion, iterative neighbourhood blending, a "min 2 predictions" rule, and betweenness-centrality edge removal. A GBDT meta-model over pairwise and graph features gained about 3 points on its own in the 3rd-place write-up. SIGMOD 2022 was won by contrastive-trained transformer embeddings with a FAISS kNN, re-ranked by a symbolic string-similarity metric.

### Cited Findings
- **Shopee 1st place:**
  - Models: 2 eca_nfnet_l1 image models + 5 text models (XLM-R, Indonesian BERT), ArcFace (margin 0.8-1.0 image, 0.6-0.8 text).
  - **Iterative Neighbourhood Blending:** kNN with k=51 and cosine thresholds; neighbour embeddings are blended into each item's embedding repeatedly.
  - 10+ tuned thresholds.
  — [Kashiwagi blog recap](https://masatakashiwagi.com/blog/kaggle-shopee-solution/); original: [Kaggle 1st place](https://www.kaggle.com/competitions/shopee-product-matching/writeups/upstage-making-ai-beneficial-1st-place-solution-fr)
- **Shopee 2nd place:**
  - Two stages: metric learning, then a meta-model classifier.
  - Graph features: top-K cosine statistics (K=5/10/15/30), PageRank, Levenshtein.
  - Sample weight 1/size^0.4.
  - Reciprocal-edge validation; recursive betweenness-centrality edge removal.
  — [Kashiwagi blog](https://masatakashiwagi.com/blog/kaggle-shopee-solution/)
- **Shopee 3rd place:** triplet loss; GBM meta-model with about 500 hand-crafted features (pairwise distances, neighbourhood density, ranks) scored **LB 0.76+**; agglomerative clustering raised it to **about 0.79** — [Kashiwagi blog](https://masatakashiwagi.com/blog/kaggle-shopee-solution/)
- **Shopee 4th place:** squared-cosine similarity fusion. Post-processing tricks: rank-2 mutual matching (bidirectional edge required), inclusion rule based on the rank-2/rank-3 difference, query expansion, and orphan re-match (items with no match get a second pass) — [Kashiwagi blog](https://masatakashiwagi.com/blog/kaggle-shopee-solution/)
- Shopee common tricks: a "minimum 2" rule (every item matches itself and at least one more, taken from the label-group property); per-model threshold tuning is a key lever — [Kashiwagi blog](https://masatakashiwagi.com/blog/kaggle-shopee-solution/)
- **SIGMOD 2022 contest (blocking):**
  - Datasets: X1 (notebook specs, about 1M rows, candidate budget 1M pairs) and X2 (Altosight product specs, about 1M rows, budget 2M pairs).
  - Metric: recall of true matches inside the fixed-size candidate set; runtime breaks ties. Training labels were transitively closed.
  — [transactional.blog archive](https://transactional.blog/sigmod-contest/2022); [contest site](https://dbgroup.ing.unimore.it/sigmod22contest/)
- **SIGMOD 2022 winner (WBSG, Brinkmann & Peeters, Mannheim):** transformer pre-trained with supervised contrastive learning, FAISS NN search in embedding space, then re-ranking of retrieved pairs with a symbolic similarity metric — [Uni Mannheim news](https://www.uni-mannheim.de/en/news/wbsg-wins-sigmod-programming-contest-2022/)
- A cross-contest lessons paper exists (SIGMOD 2020-2022 ER contests, SIGMOD Record 2023). It was not retrievable (403) — [ACM](https://dl.acm.org/doi/10.1145/3615952.3615965)

### Inferences
- Tricks we can transfer cheaply to F0.5 per query: mutual/reciprocal best-match rules, a rank-gap (top-1 minus top-2 score) feature or rule, query expansion via S2<->S3 edges, and "at least one prediction" when top-1 is confident, since our EDA says only 5.6% of queries are singletons.
- Graph and neighbourhood-statistic features (rank of the candidate, score gap to the best, number of candidates above a threshold) worked in Shopee's 2nd and 3rd place solutions and in Foursquare. They cost little to add to a GBDT.

### Gaps
- Exact Shopee top private F1 (I believe about 0.79 range, unverified beyond the 3rd-place figure). SIGMOD 2022 final recall numbers. SIGMOD 2020/2021 winning F-measures. No AMLC prior-year ER task found. No dedicated "company name matching" Kaggle competition found.

## Q3. Benchmarks: how close to 0.99 do standard systems get?

### Takeaway
On public ER benchmarks, clean structured data (DBLP-ACM, Fodors-Zagats) reaches about 0.99-1.00 F1. Product data (Abt-Buy, Walmart-Amazon, Amazon-Google, WDC Products) sits at about 0.75-0.96. Company/employer matching with Ditto reached about 0.965. Only low-ambiguity data with regular noise gets to about 0.99. Hard corner cases (near-duplicate non-matches) cap real product data near 0.80.

### Cited Findings
- **Ditto F1, DeepMatcher+ -> Ditto:**

  | Dataset | DeepMatcher+ | Ditto |
  |---|---|---|
  | Amazon-Google | 70.7 | 75.58 |
  | Beer | 78.8 | 94.37 |
  | DBLP-ACM | 98.45 | 98.99 |
  | DBLP-Scholar | 94.7 | 95.6 |
  | Fodors-Zagats | 100 | 100 |
  | iTunes-Amazon | 91.2 | 97.06 |
  | Walmart-Amazon | 73.6 | 86.76 |
  | Dirty Walmart-Amazon | 53.8 | 85.69 |
  | Dirty iTunes-Amazon | 79.4 | 95.65 |
  | Abt-Buy | 62.8 | 89.33 |
  | Company | 92.7 | 93.85 |

  Optimisation gains: domain knowledge "up to 8%"; augmentation +1.39% (ER-Magellan) and +2.53% (WDC); summarisation took Company from 41% to 93%+.
  — [Ditto, Li et al. VLDB 2021](https://ar5iv.labs.arxiv.org/html/2004.00584)
- **Ditto real-world employer/company matching:** 789k internal vs 412k public records; 10.6M candidate pairs after blocking; 20k labelled pairs (39% positive). **F1 96.53%.** Sentence-BERT blocking gave a 3.8x end-to-end speedup — [Ditto paper](https://ar5iv.labs.arxiv.org/html/2004.00584)
- **Peeters et al., LLM entity matching (arXiv 2310.11244, latest version), zero-shot F1:**

  | Dataset | GPT-4 | Llama-3.1 | Mixtral |
  |---|---|---|---|
  | WDC Products | 89.61 | 83.67 | 53.37 |
  | Abt-Buy | 95.78 | 89.84 | 82.20 |
  | Walmart-Amazon | 89.67 | 84.85 | 70.44 |
  | Amazon-Google | 76.38 | 73.99 | 40.98 |
  | DBLP-Scholar | 89.82 | 86.32 | 77.75 |
  | DBLP-ACM | 98.41 | 98.81 | 90.32 |

  — [Peeters et al.](https://arxiv.org/html/2310.11244)
- **Same paper, fine-tuned PLM baselines (RoBERTa / Ditto):**

  | Dataset | RoBERTa | Ditto |
  |---|---|---|
  | WDC Products | 77.53 | 84.90 |
  | Abt-Buy | 91.21 | 91.31 |
  | Walmart-Amazon | 87.02 | 86.39 |
  | Amazon-Google | 79.27 | 80.07 |
  | DBLP-Scholar | 93.88 | 94.31 |
  | DBLP-ACM | 99.14 | 99.00 |

  Few-shot averages: GPT-4 90.8 (10-shot); Llama-3.1 85.85. PLMs lose 22-61 F1 points on unseen entities — [Peeters et al.](https://arxiv.org/html/2310.11244)
- **WDC Products benchmark:** with 80% corner cases, the best systems (R-SupCon, Ditto, RoBERTa) reach only **72.18-79.99 F1**. All systems struggle with unseen entities — [WDC Products paper](https://arxiv.org/html/2301.09521)

### Inferences
- A small open LLM (Llama-class, ≤8B) zero-shot is roughly 10+ F1 points worse than GPT-4 and worse than a fine-tuned PLM or GBDT. On CPU within 48h it is unlikely to beat a well-featured GBDT. Use it at most as a re-ranker for a small ambiguous subset.
- 0.989 macro F0.5 is above almost every real-data public benchmark except the cleanest (DBLP-ACM / Fodors-Zagats). That suggests the AMLC data is generated with learnable corruption patterns, or that most queries are easy and the top teams handle the hard tail (decoys, singletons) well.

### Gaps
- No OpenSanctions or company-specific benchmark tables were retrieved. I found no Magellan (random forest) numbers beyond the statement that DeepMatcher mostly beats Magellan — [Ditto/ACM survey](https://dl.acm.org/doi/fullHtml/10.1145/3431816).

## Q4. Synthetic-noise ER datasets (FEBRL / GeCo) and why near-perfect scores happen

### Takeaway
FEBRL and GeCo generate records by duplicating clean originals and applying a finite, parameterised set of corruptions: keyboard typos, OCR errors, phonetic substitutions, common misspellings, missing values and swaps. Because the corruption space is small and systematic, features that model each corruption type separate matches from non-matches almost perfectly. I found no citable study quantifying "near-perfect F1 on synthetic data".

### Cited Findings
- GeCo generates personal records with configurable corruptions that mimic OCR errors, keyboard errors, phonetic errors and common misspellings — [GeCo (ResearchGate)](https://www.researchgate.net/publication/258631853_GeCo_an_online_personal_data_generator_and_corruptor)
- The FEBRL generator makes original records, then duplicates them with controlled corruption and keeps the original-duplicate links. FEBRL2 = 5,000 records (4,000 originals + 1,000 duplicates); FEBRL3 = 5,000 (2,000 originals + 3,000 duplicates) — [FEBRL (ResearchGate)](https://www.researchgate.net/publication/220272032_Febrl_-_An_open_source_data_cleaning_deduplication_and_record_linkage_system_with_a_graphical_user_interface)

### Inferences
- If AMLC data is generated this way (our EDA suggests a generator: each S2/S3 record maps to at most one S1, 5.6% singletons, Indic scripts, a France-heavy test set), look for generator artifacts:
  - which corruption operators were applied: abbreviation of legal suffixes (Pvt Ltd / Private Limited, SARL, Inc), transliteration to Indic script, token drop or swap, typos, address reformatting;
  - how decoys (hard negatives) are built, e.g. same name with a different city, or a sibling branch with a different street number.
- Explicit features per operator (for example "names equal after legal-suffix normalisation plus transliteration", "street number equal", "postcode equal") can push a GBDT toward about 0.99.
- One cheap artifact exploit: the one-to-one constraint (each S2/S3 record matches at most one S1). Solve it as an assignment problem, where each S2/S3 record is assigned only to its highest-scoring S1 query. This removes false positives shared between queries.

### Gaps
- No source found on a "synthetic business record generator" or on published leakage/artifact analyses of synthetic ER datasets.

## Q5. Typical score progression: blocking -> matcher -> post-processing

### Takeaway
Across sources the pattern repeats. Blocking at about 25-50 candidates gets about a 0.977-0.99 ceiling. A first GBDT reaches roughly 0.90-0.93. Richer features and transformer ensembles add about 0.01-0.02. Graph or transitive post-processing adds another roughly 0.01-0.04 and can raise the ceiling itself.

### Cited Findings
- Foursquare 7th place: candidate ceiling 0.9778 with 32 candidates, 0.9935 with post-processing — [Future Architect](https://future-architect.github.io/articles/20220720a/)
- Foursquare, listed 2nd: 0.907 -> 0.946 across stages including GNN post-processing — [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- hobbitlab: 0.922 -> 0.928 (+2nd encoder) -> 0.931 (+GBDT features) -> 0.940 (null filling + 4-model fusion) — [hobbitlab README](https://github.com/hobbitlab/Foursquare-Location-Matching/blob/main/readme.md)
- Shopee 3rd: GBM meta-model 0.76+ -> about 0.79 with clustering post-processing — [Kashiwagi blog](https://masatakashiwagi.com/blog/kaggle-shopee-solution/)
- Ditto: domain-knowledge injection up to +8%; augmentation +1.4-2.5% — [Ditto](https://ar5iv.labs.arxiv.org/html/2004.00584)

### Inferences
- For us (0.87 now, 0.975 ceiling), most of the remaining gain should come from three things. First, a better matcher: a GBDT with rank/gap/relative features within each query group, not only absolute similarities. Second, per-query decision rules tuned for F0.5: a relative threshold against top-1, top-1 forced when confident, reciprocal-best checks. Third, S2<->S3 transitive/graph consistency and one-to-one assignment. Every top write-up relies on this "listwise context" (rank within candidates, score gap), and it is cheap to compute on CPU.

### Gaps
- No clean per-stage ablation from the actual Foursquare winners (write-ups not retrievable).
