# Blocking / Candidate Generation for Entity Resolution at Scale

Scope: blocking families and their reported recall vs candidate-set-size trade-offs; how to keep candidate sets small; Sparkly and TF-IDF/BM25 blocking; libraries (license, CPU feasibility); multi-pass blocking; scaling numbers. Written for AMLC 2026 Business ER (S1 = 1.7M deduped reference, S2+S3 about 10M, US/India/France, macro F0.5 per S1, smaller candidate sets ranked higher, CPU-only 16 GB RAM, 48 h).

Terminology: PC (pair completeness) = recall of the blocker; PQ (pairs quality) = precision of the candidate set; RR (reduction ratio) = 1 - |C| / |A x B|; CSSR (candidate set size ratio) = |C| / |A x B|.

## 1. Blocking families and their reported recall vs reduction / candidates-per-record

### Takeaway
Across the two largest recent head-to-head benchmarks (Sparkly VLDB 2023; Papadakis et al. filtering benchmark 2022), **top-k (cardinality-based) nearest-neighbour search over syntactic char-q-gram TF-IDF vectors is the most predictable, best-or-near-best blocker**. It beats self-supervised deep-learning blockers (DeepBlocker), LSH/MinHash and similarity-threshold joins on candidate size at equal recall. Tuned JedAI token-blocking workflows with meta-blocking can match it but are unpredictable. The data do not support moving to dense or LSH blocking in 48 h.

### Cited Findings
**Families, as a taxonomy**
- The filtering benchmark splits methods into (i) blocking workflows, which group profiles by identical or similar signatures (standard/token blocking, q-grams, extended q-grams, suffix arrays), followed by block purging/filtering and comparison cleaning (meta-blocking); and (ii) nearest-neighbour methods, which vectorize profiles and return the closest ones per query, using sparse vectors (epsilon-join, kNN-join) or dense vectors (FAISS, SCANN, LSH variants, DeepBlocker) — [Papadakis et al., Benchmarking Filtering Techniques for ER, arXiv 2202.12521](https://arxiv.org/pdf/2202.12521)
- Survey reference for the full taxonomy (blocking, sorted neighbourhood, canopy, meta-blocking, filtering): [Papadakis, Skoutas, Thanos, Palpanas, "Blocking and Filtering Techniques for ER: A Survey", ACM CSUR 53(2) 2020, arXiv 1905.06167](https://arxiv.org/pdf/1905.06167)

**Sparkly (TF-IDF/BM25 top-k via Lucene), recall and output size at fixed k** ([Paulsen, Govind, Doan, VLDB 16(6) 2023](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf))
- Method: index the smaller table A. For each tuple of B, return the top-k tuples of A by BM25 score over the concatenated blocking attributes, tokenized to **character 3-grams**. Output size is capped at k·|B|.
- Recall across 15 benchmark datasets: **92.5-100% at k=10, 96.4-100% at k=20, 98.7-100% at k=50**. By comparison, JedAI's PBW reached 74.5-100% recall with output as large as 4.2 billion pairs (Songs). DBW reached 84.7-99.9% with output up to 454.5M. JD produced small output but only 35.4-96.4% recall. Union(DeepBlocker, RBB) gave 83-99.9%.
- Example rows (|C|, recall): Abt-Buy textual: Sparkly k=10 gives 10.9k pairs at 98.1%, while JD gives 1.2k at 65.2% and PBW 4.7k at 74.5%. Amazon-Google structured: k=10 gives 33.3k at 96.8%, and k=50 gives 165.9k at 99.2%.
- Against the two best DL blockers (Autoencoder and Hybrid from DeepBlocker), plotted as recall vs CSSR, Sparkly Manual has lower CSSR at every recall on all 15 datasets. On Amazon-Google at 98% recall, Sparkly's CSSR is 2.5% against 10% for the DL blockers. The gap widens as recall approaches 100%.
- Against kNN-cosine on 5-grams (the blocker recommended by the JedAI group): Sparkly is better on 10 datasets, comparable on 1 and slightly worse on 4. **kNN-cosine on 3-grams is about equal to 5-grams, and both beat Jaccard.**
- On large data, recall@10/25/50 for Sparkly Manual/Auto: MusicBrainz 20M gives 79/95, 87/97, 91/98. BC 2.5M gives 99/79, 100/89, 100/94. **Autoencoder on BC 2.5M: 30/35/40; Hybrid: 81/84/85.**

**DeepBlocker (self-supervised DL blocking)** ([Thirumuruganathan et al., VLDB 14(11) 2021](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf))
- Of 8 DL designs, Autoencoder (fastText embeddings, SIF aggregation, feed-forward autoencoder, FAISS top-k cosine) is best on structured and dirty data. Hybrid (Autoencoder plus cross-tuple training) is best on textual data. All 8 exceed 90% recall "with relatively small candidate sets".
- DL beats the industrial rule-based blocker RBB on recall for all but one dataset, especially textual and dirty ones. On Hospital and Songs, DL's candidate set is much larger than RBB's at comparable recall.
- **Union of DL and RBB**: recall rises by 0.3-6.7 points absolute on structured data, 8.5-14.5 on textual and 0.2-9.9 on dirty. The candidate set grows by at most 49.9%, 57.6% and 12.3% respectively, and "in most cases, the actual size increase is minimal". This supports unioning blockers that make different errors.

**Filtering benchmark (Papadakis et al. 2022), target PC ≥ 0.9, tuned configs**
- **Cardinality (top-k) thresholds beat similarity thresholds.** kNN-Join outperforms epsilon-Join on 8 datasets (average rank 3.2 vs 5.7), because a cardinality threshold "operate[s] locally, selecting the best candidates per query entity". The best similarity-based method (epsilon-Join) still cuts brute-force pairs by 99% — [arXiv 2202.12521](https://arxiv.org/pdf/2202.12521)
- **LSH performs poorly.** MinHash-LSH, cross-polytope LSH and hyperplane LSH reach high recall only by producing huge candidate sets: they reduce brute-force pairs by just 48%, 91% and 89% on average, consistently worse than cardinality-based NN methods — [same](https://arxiv.org/pdf/2202.12521)
- **The best default sparse configuration** is kNN-Join with cosine similarity on the character 5-gram multiset (C5GM). It cleans values (stop-word removal and stemming), queries with the **smaller** dataset and uses **K=5**. Its recall still falls to 0.8-0.9 on 3 datasets — [same](https://arxiv.org/pdf/2202.12521)
- **Syntactic representations beat semantic (pre-trained embedding) ones** on all but 2 datasets. Semantic vectors add false positives because of out-of-vocabulary, domain-specific terms — [same](https://arxiv.org/pdf/2202.12521)
- **Among blocking workflows, tuned Standard (token) Blocking + meta-blocking (SBW)** ranks best overall on precision (average rank 2.2, best on the 8 largest datasets). The parameter-free PBW ranks worst (11.9), so tuning matters a lot. Using q-grams or suffix arrays as signatures "increases significantly the number of candidate pairs, without any benefit in recall" compared with whole-token signatures, in schema-agnostic settings — [same](https://arxiv.org/pdf/2202.12521)
- The authors still prefer kNN-Join because (i) its candidate count is linear in input size and (ii) it is easy to configure — [same](https://arxiv.org/pdf/2202.12521)

**Meta-blocking (block-graph pruning)**
- Meta-blocking builds a graph from a redundant block collection (e.g., token blocking), weights each edge by co-occurrence schemes (CBS, ECBS, ARCS, JS, EJS), and prunes with WEP/CEP (edge-centric) or WNP/CNP (node-centric, i.e., per-entity top-k). Reciprocal variants keep an edge only if it survives in both endpoint neighbourhoods — [Papadakis et al., Meta-Blocking, TKDE 2014](https://helios2.mi.parisdescartes.fr/~themisp/publications/tkde13-metablocking.pdf); [Enhanced meta-blocking, EDBT 2016](https://openproceedings.org/2016/conf/edbt/paper-193.pdf)
- Search-result summaries of those papers report block-cardinality reductions of 64-75% with recall falling by less than 0.5% in most cases, and Refined CNP keeping 18% fewer comparisons than CNP at the same recall. These figures come from a search snippet, not a verified table — [EDBT 2016](https://openproceedings.org/2016/conf/edbt/paper-193.pdf)
- Supervised meta-blocking (GSM) turns several weighting schemes into a feature vector for a binary classifier that decides which candidate edges to keep, and reports that very small training sets suffice. I read only the abstract, so no numbers are given here — [Generalized Supervised Meta-blocking, arXiv 2204.08801](https://arxiv.org/pdf/2204.08801)

**Supervised dense blocking**
- SC-Block (supervised contrastive embeddings plus NN search) yields smaller candidate sets than 8 state-of-the-art blockers at 99.5% pair completeness. End-to-end pipelines run 1.5-2x faster at the same F1, and up to 8x faster (2.5 h down to 18 min) on a large product benchmark, after 5 minutes of training — [Brinkmann et al., SC-Block, arXiv 2303.03132](https://arxiv.org/abs/2303.03132)

### Inferences
- The current pipeline (char-3gram TF-IDF top-K with sparse_dot_topn, cosine) is essentially the "kNN-cosine-3gram" and "TFIDF-cosine" configuration that both benchmarks rank at or near the top. The remaining gains are in tuning (K, direction, fields, unions, cleaning), not in switching family.
- LSH/MinHash is not worth building: it is inferior in both benchmarks.
- Dense embeddings (DeepBlocker-style, or a small multilingual encoder) give an **orthogonal** candidate source. The DeepBlocker union result suggests that a small top-k from a second, different representation can buy recall cheaply. On a CPU it is only worth it if a small model can embed about 12M strings in a few hours, which is plausible for a MiniLM-sized encoder, but I found no measured figure.
- Given the EDA fact that each S2/S3 record links to at most one S1 record, and that S1 is the smaller, deduplicated side: querying **from S2/S3 into an S1 index with very small k (1-3)** is the direction the benchmarks recommend. They use the smaller table as index and query with the other in Sparkly; the filtering benchmark queries with the smaller set to minimize pairs; in our case the constraint argues for the reverse. It naturally caps output at about k × 10M pairs, and each S2/S3 record can contribute to at most one S1 anyway.

### Gaps
- No benchmark covers multilingual or mixed-script business names (Hindi, Telugu, Tamil). All reported numbers come from English product, citation and music datasets.
- I could not verify meta-blocking PC/RR tables directly; the figures above come from search-result summaries.
- The CSUR 2020 survey's aggregate numbers were not extracted.

## 2. How the best systems keep candidate sets tiny (adaptive cutoffs, top-k, learned scoring)

### Takeaway
The consistent lesson is **per-record (local) cardinality cuts beat global similarity thresholds**, reciprocal or node-centric pruning removes more pairs at almost no recall cost, and a cheap learned scorer applied before the cut (supervised meta-blocking, GSM) is the standard way to shrink sets further. Smaller candidate sets were shown to preserve matching F1 while speeding pipelines up (SC-Block).

### Cited Findings
- **Top-k is critical; thresholding fails.** In Sparkly's data, gold-match similarity scores (tf-idf, Jaccard, cosine) are spread across [0,1] on 2 of the 3 datasets shown. A global threshold therefore loses recall unless it is set so low that the output explodes, whereas top-k gets high recall "without blowing up the output size" — [Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)
- **IDF matters; TF barely does for short fields.** Removing idf hurts badly. Removing tf gives no clear change on short attributes, but on long textual records recall@50 drops from 62% to 33% — [Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)
- **Symmetric TF-IDF cosine beats plain BM25** on many datasets, because BM25 weights only the document side. The authors' SM+ variant adds query-side tf-idf to BM25 and is best or near-best on all datasets. BM25 parameters: k1 between 1 and 2 barely matters; b between 0.5 and 1 changes CSSR by up to 2%, with b=0.75 a good default — [Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)
- **Tokenizer:** 3-grams are a good default. 2-grams and 4-grams perform worst, and 2-grams are also the slowest. Changing the tokenizer changes CSSR by up to 11.5% — [Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)
- **Attribute and tokenizer auto-selection (Sparkly Auto)** scores each candidate configuration on a 10K sample of B using top-250 lists. It normalizes each BM25 score curve by its maximum, ranks configurations by how "discriminative" the curve is (a steep drop after rank 1, measured via AUC), and greedily combines per-attribute BM25 scores. SA beats manual selection on 10 of 15 datasets and is at most 0.7% worse in CSSR at 98% recall elsewhere. It is also faster, since per-attribute scoring with word tokens on some fields is quicker than scoring a 3-gram concatenation — [Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)
- **Very small K works.** The filtering benchmark's default kNN-Join uses K=5, and its tuned configurations typically select "a very low number of nearest neighbors per query entity". DeepBlocker's tuned default also uses K=5 — [Filtering benchmark](https://arxiv.org/pdf/2202.12521)
- **Choosing K without full labels:** DeepBlocker increases K until MatchCatcher, applied to the 200 most similar pairs outside the candidate set, finds no more than 10 true matches — [DeepBlocker](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)
- **Node-centric and reciprocal pruning:** CNP keeps a per-entity priority queue whose size depends on the average number of blocks per entity. Reciprocal CNP/WNP keep an edge only if it passes both endpoints' criteria, the most aggressive of the pruning options — [Enhanced meta-blocking EDBT 2016](https://openproceedings.org/2016/conf/edbt/paper-193.pdf); [GSM](https://arxiv.org/pdf/2204.08801)
- **Candidate size vs downstream F1:** SC-Block's smaller candidate sets kept end-to-end F1 unchanged while cutting runtime 1.5-8x — [SC-Block](https://arxiv.org/abs/2303.03132)
- **Fine-tuning parameters "significantly raises" filtering performance for every method type**, and the authors call for automatic, label-free tuning — [Filtering benchmark](https://arxiv.org/pdf/2202.12521)

### Inferences (actionable for AMLC)
- **Reverse-direction top-1/top-2 with reciprocal check:** for each S2/S3 record, take its top-2 S1 neighbours by TF-IDF cosine on the name and address union. Keep a pair (s1, x) only if s1 is x's top-1, or s1 is x's top-2 and the score gap is small. Because each x maps to at most one S1, anything beyond x's top-2 is almost always a wrong pair. This is reciprocal CNP adapted to a 1-to-many constraint. It should cut the candidate count with little recall loss; validate on the labelled US and India data.
- **Learned scorer before the cut:** use the existing reranker as a "supervised meta-blocking" step. Features are the name 3-gram cosine, address cosine, rank of s1 in x's list, rank of x in s1's list, score gap to the next candidate, same pincode/zip, number of candidates, and field-emptiness flags. Then cut per S1 by **expected-F0.5 maximization**: sort by calibrated probability and choose the prefix that maximizes expected F0.5. F0.5 weights precision, so this favours short lists. This is the principled version of the current adaptive cut; it is my inference, not a sourced benchmark result.
- **Use max-normalized score curves (Sparkly Auto) as features:** score divided by the top-1 score for that query, plus drop-off, to decide the per-S1 k.
- Keep blocking recall@K high (K about 10-20 pre-rerank). Enforce smallness only on the final submitted candidate_pairs after reranking, since organizers review candidate_pairs.tsv.

### Gaps
- I found no paper that directly optimizes candidate-set size under an F0.5 metric.
- I found no public result quantifying "dynamic k" per record, other than node-centric pruning with an average-based threshold.

## 3. Sparkly and BM25/Lucene blocking vs deep learning blockers

### Takeaway
Sparkly shows that plain tf/idf top-k (Lucene BM25, 3-grams) beats DeepBlocker and JedAI on recall at a given candidate size, is far more predictable, and scales to 26M records in a couple of hours on a cluster. Symmetric TF-IDF cosine, which is what we use, is at least as good as BM25.

### Cited Findings
- Sparkly Manual gives 92.5-100% recall at k=10 on 15 datasets. It beats Autoencoder and Hybrid at every recall level and is better on 14 of 15 datasets even when the DL methods are restricted to the same attributes — [Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)
- DL blockers do not scale well. Autoencoder took 691 min on MB 10M versus Sparkly's 132/61 min (SM/SA), and 146 min on BC 2.5M versus 44/11 min. Hybrid ran out of memory on 5M-record datasets and took 2,719 min on BC 2.5M. DL recall at k=50 on BC 2.5M was 40% (Autoencoder) and 85% (Hybrid), against Sparkly's 100/94 — [Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)
- Lucene only became fast enough for tf/idf blocking after it adopted block-max WAND (around 2015) — [Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)
- Code: [anhaidgroup/sparkly](https://github.com/anhaidgroup/sparkly), BSD-3-Clause (GitHub API). It requires PySpark and PyLucene, which are heavy to install on Windows; the authors themselves note installation pain with alternatives.

### Inferences
- There is no need to adopt Sparkly itself. sparse_dot_topn with an L2-normalized TF-IDF matrix computes the same "TFIDF-cosine" top-k that Sparkly found at least as good as BM25.
- If BM25 is wanted as an extra signal, bm25s (MIT) is a pure-Python option. It is untested at this scale here.

### Gaps
- There is no CPU single-machine Sparkly benchmark; all runtimes are from 10-30-node AWS clusters.

## 4. Libraries: license and CPU feasibility

### Takeaway
Everything needed is permissively licensed. The two to avoid are **Zingg (AGPL-3.0)** and the transliteration packages **Unidecode (GPL-2.0)** and **libindic indic-trans (AGPL-3.0)**. Use **anyascii (ISC)** or **indic_transliteration (MIT)** for script normalization instead.

### Cited Findings (licenses from the GitHub API, 2026-09-25)
| Library | License | Role / CPU notes |
|---|---|---|
| sparse_dot_topn (ing-bank) | Apache-2.0 | Sparse matmul with top-n, OpenMP threads. `sp_matmul_topn(A, B, top_n, threshold, sort, n_threads, density, idx_dtype)`. Uses a max-heap so memory is O(top_n) per row; `zip_sp_matmul_topn` merges column-chunked results. README: up to **6x faster** than naive on two 20k x 193k TF-IDF matrices, top-10, 8 cores (Apple M2 Pro) — [GitHub](https://github.com/ing-bank/sparse_dot_topn) |
| pyJedAI | Apache-2.0 | Block building, block/comparison cleaning (meta-blocking), similarity joins and vector NN in one Python framework — [GitHub](https://github.com/AI-team-UoA/pyJedAI) |
| Splink (UK MoJ) | MIT | Fellegi-Sunter with SQL blocking rules. Multiple rules are OR-unioned and deduplicated. Advice: "a longer list of strict blocking rules" beats a short loose list. `count_comparisons_from_blocking_rule()` estimates pair counts before running — [Splink docs](https://moj-analytical-services.github.io/splink/topic_guides/blocking/blocking_rules.html) |
| dedupe | MIT | Learned blocking predicates plus active learning — [GitHub](https://github.com/dedupeio/dedupe) |
| recordlinkage | BSD-3-Clause | Standard blocking, sorted neighbourhood — [GitHub](https://github.com/J535D165/recordlinkage) |
| Zingg | **AGPL-3.0** (avoid) | Spark-based learned blocking — [GitHub](https://github.com/zinggAI/zingg) |
| FAISS | MIT | Dense ANN. The filtering benchmark calls it "by far the fastest and most scalable filtering technique" on large data (run-time grows only 700x from 10K to 2M entities) — [GitHub](https://github.com/facebookresearch/faiss); [benchmark](https://arxiv.org/pdf/2202.12521) |
| hnswlib | Apache-2.0 | Dense HNSW — [GitHub](https://github.com/nmslib/hnswlib) |
| pynndescent | BSD-2-Clause | kNN graph; supports sparse inputs — [GitHub](https://github.com/lmcinnes/pynndescent) |
| datasketch | MIT | MinHash LSH (benchmarked poorly for ER, see section 1) — [GitHub](https://github.com/ekzhu/datasketch) |
| Annoy | Apache-2.0 | Dense ANN — [GitHub](https://github.com/spotify/annoy) |
| usearch | Apache-2.0 | Dense HNSW — [GitHub](https://github.com/unum-cloud/usearch) |
| RapidFuzz | MIT | Fast string similarity for rerank features — [GitHub](https://github.com/rapidfuzz/RapidFuzz) |
| bm25s / rank_bm25 | MIT / Apache-2.0 | BM25 — [bm25s](https://github.com/xhluca/bm25s), [rank_bm25](https://github.com/dorianbrown/rank_bm25) |
| Sparkly | BSD-3-Clause | Needs Spark and PyLucene — [GitHub](https://github.com/anhaidgroup/sparkly) |
| anyascii | ISC | Any-script to ASCII transliteration — [GitHub](https://github.com/anyascii/anyascii) |
| indic_transliteration | MIT | Indic scheme transliteration — [GitHub](https://github.com/indic-transliteration/indic_transliteration_py) |
| Unidecode | **GPL-2.0** (avoid if code must be MIT/Apache) | — [GitHub](https://github.com/avian2/unidecode) |
| libindic indic-trans | **AGPL-3.0** (avoid) | — [GitHub](https://github.com/libindic/indic-trans) |

### Inferences
- The DeepBlocker repository could not be found at anhaidgroup/deepblocker (the GitHub API returned "Not Found"), so its license is unverified.
- For Indic scripts at blocking time: build char 3-grams on **both** the native-script string and an anyascii-transliterated string (two TF-IDF passes, then union). Transliteration lets "Sharma Traders" in Latin script meet its Devanagari form.

### Gaps
- The pyJedAI README fetch did not return a verified list of methods; the paper or docs should be checked if it is used.

## 5. Empty addresses, unrelated trade names, and multi-pass blocking

### Takeaway
The literature's answer is **multiple strict passes, unioned** (Splink, DeepBlocker+RBB union, Sparkly Auto's multi-attribute configurations), with schema-agnostic blocking as a robust fallback when fields are missing or misplaced.

### Cited Findings
- Splink recommends many strict rules OR-ed together to cover different failure modes, such as a typo in one field while another field agrees — [Splink](https://moj-analytical-services.github.io/splink/topic_guides/blocking/blocking_rules.html)
- A union of two different blockers (DL and rules) raised recall by up to 14.5 points while enlarging candidate sets by 12-58% at most, usually minimally — [DeepBlocker](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)
- Schema-agnostic settings (all attribute values pooled) give "robust effectiveness, as they inherently address heterogeneous schemata as well as misplaced and missing values". Schema-based settings are faster but less stable — [Filtering benchmark](https://arxiv.org/pdf/2202.12521)
- Sparkly Auto combines per-attribute BM25 scores (sum over up to 3 attribute-tokenizer pairs), so that no single field dominates — [Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)

### Inferences (actionable)
- Suggested passes, each producing a small top-k:
  - (a) name 3-gram cosine, per country;
  - (b) address word cosine;
  - (c) a **schema-agnostic concatenation** of name, address and city 3-grams, which catches cases where the trade name differs from the legal name but the address matches, or where the address is empty but the name plus city matches;
  - (d) a transliterated-name pass for India;
  - (e) a cheap exact-key pass, such as phone or pincode plus the first name token, if those fields exist.
- For records with an empty address, skip pass (b) and raise k in (a) and (c) slightly. Emptiness should be a rerank feature, not a reason to drop the record.
- **Transitive expansion:** because S1 is deduplicated and each S2/S3 maps to at most one S1, expansion through other records applies mainly between S2 and S3. If x2 in S2 and x3 in S3 are near-duplicates (a cheap S2-S3 top-1 join), and x2 is strongly linked to s1, add (s1, x3) as a candidate. This is an inference; it is not benchmarked in the sources.

### Gaps
- No source quantifies multi-pass recall gains specifically for business names or addresses.

## 6. Scaling numbers (records/sec on CPU for TF-IDF top-k)

### Takeaway
Published numbers are cluster-based. Extrapolated to our machine, the order of magnitude is roughly millions of top-k queries per hour. Measure locally on a 100K-row chunk and extrapolate.

### Cited Findings
- Sparkly Auto blocks 10M x 10M in under 100 min on 10 AWS m5.4xlarge nodes (160 cores), about $12.5. WDC 26M took 130 min (SA) or 603 min (SM) on 30 nodes. MusicBrainz 20M took 168 or 449 min. Indexing 10M records took 115 s and the index was 1.3-2.0 GB. Scaling was slightly superlinear — [Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)
- In the filtering benchmark, FAISS processed the 2M-entity synthetic dataset in about 1.3 h, while blocking workflows needed about 3.2 h. Only blocking workflows, FAISS and SCANN scaled to 2M records within 4 h. Run-times grew superlinearly (more than 200x) but subquadratically (less than 40,000x) from 10K to 2M records — [Filtering benchmark](https://arxiv.org/pdf/2202.12521)
- sparse_dot_topn: up to 6x speedup on 20k x 193k matrices at top-10 on 8 cores; no absolute rows/sec figure is published in the README — [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn)

### Inferences
- A rough CPU extrapolation from Sparkly: 160 cores times 100 min is about 16,000 core-minutes for 10M queries, or about 1.6 ms per query-core with Lucene WAND on long concatenated strings. With 18 threads that is roughly 15 h for 10M queries — too slow for the long-string, full-concat case.
- Per-country partitioning (US / India / France) and short name-only fields reduce cost a lot. Sparkly notes that short attributes are much faster, and sparse_dot_topn on short name 3-gram vectors is typically far faster than Lucene on concatenations.
- Memory: run chunked queries (e.g., 50-200K rows per chunk) against the S1 matrix. Use `idx_dtype=int32` and float32 data, and set `density` low to avoid over-allocation within 16 GB.

### Gaps
- I found no published single-machine records/sec figure for sparse_dot_topn on millions of short strings; this has to be measured locally.
