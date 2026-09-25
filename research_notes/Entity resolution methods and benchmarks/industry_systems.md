# Industrial / Commercial / Open-Source Entity Resolution Systems for Business & Place Records

Scope note: researched 2026-09-25 for AMLC 2026 (3-source business ER, names+addresses, US/India/France, macro F0.5, CPU-only, ~48h). Only design lessons are relevant; no services may be called. About 20 tool calls were used, so coverage is uneven: Yelp, Senzing, Splink, OpenSanctions, AWS ER, Foursquare/Overture and the Amazon GNN paper are covered well. Google, LinkedIn/Microsoft, Tamr, Zingg and D&B were **not** verified from sources this session (see Gaps).

## Q1. Amazon: publications and AWS Entity Resolution design

### Takeaway
AWS Entity Resolution offers two kinds of workflow. Rule-based workflows use explicit match keys with Exact or Fuzzy comparison (Levenshtein, Cosine, Soundex thresholds) combined with AND/OR. ML workflows use a proprietary Amazon model with per-record confidence scores and split thresholds (a strict one for automatic merges, a looser one for review). Amazon Science's product-ER work focuses on detecting bad clusters after pairwise matching, because pairwise models alone over-merge.

### Cited Findings
- AWS ER rule-based workflows come in two types. "Simple" supports only Exact matching with AND. "Advanced" supports Exact and Fuzzy matching with AND, OR and parentheses, and works in batch and incremental modes but not real-time. — [AWS docs: rule-based workflow](https://docs.aws.amazon.com/entityresolution/latest/userguide/creating-matching-workflow-rule-based.html); [Advanced rule type](https://docs.aws.amazon.com/entityresolution/latest/userguide/rule-based-mw-advanced.html)
- Advanced rules can put similarity, distance or phonetic thresholds on string fields using Levenshtein distance, cosine similarity and Soundex, "without custom pre-processing". — [AWS ER features / search summary](https://aws.amazon.com/entity-resolution/features/)
- The ML workflow uses an AWS-provided model that handles variations in Name, Address, Email, Phone and DOB. It is aimed at person/customer data, not specifically businesses. — [AWS ER FAQ](https://aws.amazon.com/entity-resolution/faqs)
- Sep 2026: AWS added record-level confidence scores to ML matching. The stated use is "differentiated thresholds": a higher confidence for automated merges and a lower one to include more records. — [AWS What's New 2026-09](https://aws.amazon.com/about-aws/whats-new/2026/09/entity-resolution-record-confidence/)
- AWS has a blog post on measuring the accuracy of rule-based vs ML matching. I did not read it in full. — [AWS for Industries blog](https://aws.amazon.com/blogs/industries/measuring-the-accuracy-of-rule-or-ml-based-matching-in-aws-entity-resolution/)
- Amazon Science, "Graph Neural Networks for Inconsistent Cluster Detection in Incremental Entity Resolution" (arXiv 2105.05957):
  - Product catalog clusters are built from pairwise matching models and then contain errors.
  - The paper treats "inconsistent cluster" detection as supervised graph classification on a weighted graph, with pairwise similarities as edge weights, using Message Passing NNs with a new aggregation scheme.
  - GNNs beat conventional methods on synthetic, public and internal data.
  - The paper notes that a 10-member cluster with one wrong member produces only 9 bad pairs out of 45, so pairwise metrics hide cluster errors.
  - [arXiv abstract](https://arxiv.org/abs/2105.05957); [Amazon Science page](https://www.amazon.science/publications/graph-neural-networks-for-inconsistent-cluster-detection-in-incremental-entity-resolution)

### Inferences
- The AWS pattern to copy is a **cascade of rules from strict to fuzzy, then a model score with two thresholds**. In our setting: an exact normalized name + same postcode rule gives an automatic match, a model score above a high threshold gives a match, and anything in the grey zone is rejected, because F0.5 favors precision.
- The Amazon GNN paper's lesson is to **check clusters after pairwise scoring**. A cheap stand-in for a GNN is to flag clusters where the average or minimum within-cluster pairwise score is low, or where one weak edge is the only link between two dense groups, and split them.

### Gaps
- I found no public document describing how AWS ER's ML model normalizes business names or addresses (legal suffixes, abbreviations). It appears to be proprietary.
- No "AutoER" Amazon paper was located in this session.

## Q2. Other systems (Yelp, Foursquare/Overture, Meta, Senzing, Splink, OpenSanctions)

### Takeaway
Business dedup systems that publish numbers run at very high precision and accept low recall. Yelp's auto-merge ran at 99.1% precision and 27.7% recall. The recurring design across these systems is:
1. Retrieval or blocking (search index, geo cells, several blocking keys).
2. A pairwise model on string, geo and phone features.
3. Clustering with explicit guards against generic values and bridging edges.

### Cited Findings
**Yelp (the closest public analogue to our task)**
- The Business Match service normalizes name, location and phone, then queries Elasticsearch with a subquery per component (TF-IDF name match, geo-distance, phone). Elasticsearch supplies candidates. A pointwise learning-to-rank model then rescores them using the subquery scores as features. This raised F1 from 91% to 95%. — [Yelp: Learning to Rank for Business Matching](https://engineeringblog.yelp.com/amp/2014/12/learning-to-rank-for-business-matching.html)
- The dedup classifier is a scikit-learn Random Forest. Features:
  - Geographic distance.
  - Synonym-aware field matching.
  - Edit distance and Jaccard on text fields.
  - An NER sub-model that detects person-named businesses (lawyers, doctors, agents), to separate a professional from their practice.
  - A logistic-regression word-alignment model on business names that finds discriminative shared and differing tokens.
  - [Yelp: Seeing Double](https://engineeringblog.yelp.com/2015/05/seeing-double-on-yelp.html)
- Yelp optimized **F0.1** (precision weighted 10x):

  | Configuration | F0.1 | Precision | Recall |
  |---|---|---|---|
  | Exact-match baseline | 0.915 | 97.1% | 13.4% |
  | Basic features | 0.9415 | 96.6% | 26.4% |
  | Full model | 0.966 | 99.1% | 27.7% |

  More than 500k duplicates were merged. — [Yelp: Seeing Double](https://engineeringblog.yelp.com/2015/05/seeing-double-on-yelp.html)
- Candidate search uses a deliberately "relatively low confidence threshold", and the strict classifier then decides. — same source.

**Foursquare / Overture (POI conflation)**
- Foursquare runs a "resolve model": a pairwise POI similarity model retrained on multi-country, multilingual, human-labeled pairs. It first finds potential matches in the existing database, then scores them. — [Foursquare Technical Guide Part 2](https://foursquare.com/resources/blog/products/technical-guide-to-foursquare-places-part-2-how-does-foursquare-get-location-data-right/)
- Foursquare Kaggle location-matching winners used:
  - Candidates from geographic proximity plus TF-IDF.
  - Filtering stages.
  - Levenshtein and Jaro-Winkler features, BERT-based models, and LightGBM/XGBoost.
  - GNN post-processing and ArcFace metric-learning embeddings.
  - [Foursquare: Finding the Right POI Match](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- Overture Places uses ML matching plus clustering. Labels are generated by LLMs with an "evidence hierarchy": functional identity, then phone, then name uniqueness and distance, then supporting signals (website, category, brand). Two independent models label each pair. Agreements are accepted and disagreements go to human review. — [Overture Places guide / release notes](https://docs.overturemaps.org/blog/2025/05/21/release-notes/)
- Overture's stated aim is "consistent conflation", meaning IDs stay stable across releases. Inconsistent conflation causes entities to split or merge as sources shift. — [Overture Places guide](https://docs.overturemaps.org/guides/places/)
- Hobby Overture conflation (not industrial, but concrete):
  - Blocking on H3 res-7 cells (about 5 km²).
  - Cascading rules: exact name gave a 31% match rate; adding Jaro-Winkler on name+address gave 68%; embeddings (mxbai-embed-large, cosine ≥ 0.87) on the residue brought the combined total to 80%.
  - Embeddings took about 20 min for 56k places, versus 0.4 s for the SQL rules.
  - [dbreunig blog](https://www.dbreunig.com/2024/09/27/conflating-overture-points-of-interests-with-duckdb-ollama-and-more.html)

**Meta/Facebook Places**
- The WWW 2014 paper "Deduplicating a places database" (Dalvi et al.) is used in production. It targets places that have only a name and location, where phone and full address are missing. Its example of the core difficulty is "Newpark Mall Gap Outlet" vs "Newpark Mall Sears Outlet": high string similarity, but different entities. — [ACM DL](https://dl.acm.org/doi/10.1145/2566486.2568034) (abstract via search snippet; full text 403)
- Meta patents describe three steps: suppression of low-quality places, deduplication/clustering, and best-page (canonical) selection. — [US patent 10282434](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10282434)

**Senzing (principle-based, entity-centric)**
- Each attribute gets three behaviors:
  - Frequency: F1 (one entity), FF (few), FM (many), FVM (very many), or NAME. ADDRESS and PHONE are FF; TAX_ID is F1.
  - Exclusivity: whether a differing value should keep records apart.
  - Stability.
  - [Senzing Principle-Based ER PDF](https://senzing.com/wp-content/uploads/Principle-Based-Entity-Resolution-092519.pdf)
- **Generic value detection**: when a value expected to be rare (e.g. an SSN) shows up on many entities, it is labelled generic and all prior records carrying it are re-evaluated. Thresholds are configurable (`setGenericThreshold`). — same PDF; [Senzing help](https://senzing.zendesk.com/hc/en-us/articles/231726307-Principle-Based-Entity-Resolution)
- About 35 principles. An incoming record is scored against **candidate entities** (entity-centric resolution) and gets one of three outcomes: MATCH, POSSIBLE MATCH or POSSIBLY RELATED. Examples:
  - `SAME_F1+CLOSE_NAME` matches unless exclusive attributes conflict.
  - `MULT_FF+CLOSE_NAME` requires **multiple** few-frequency attributes (e.g. address AND phone) plus a close name.
  - Organization examples: "ABC Company" vs "ABC Enterprises Inc." at the same address with different tax IDs is only a possible match. "ABC Company" vs "XYZ Enterprises Inc." at the same address and phone is possibly related, not a match.
  - [Senzing PDF](https://senzing.com/wp-content/uploads/Principle-Based-Entity-Resolution-092519.pdf)

**Splink (UK MoJ, Fellegi-Sunter)**
- Term-frequency adjustment: adjusted match weight = base weight + a TF adjustment that is inversely related to how often the value occurs. Matching on a rare value counts for more than matching on a common one ("Portia" vs "Smith").
  - For fuzzy (non-identical) matches it uses the **higher** of the two term frequencies, which is the conservative choice.
  - `tf_minimum_u_value` caps the boost from very rare values.
  - Apply TF to a derived column to avoid double-penalizing composite names.
  - [Splink TF guide](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html)
- Blocking: a list of rules OR'd together, with duplicate pairs removed. The recommendation is "a longer list of strict blocking rules" rather than a few loose ones, and to count comparisons per rule before running. — [Splink blocking guide](https://moj-analytical-services.github.io/splink/topic_guides/blocking/blocking_rules.html)
- Clustering is connected components on edges above a probability threshold (`cluster_pairwise_predictions_at_threshold`). — [Splink clustering docs](https://moj-analytical-services.github.io/splink/api_docs/clustering.html)

**OpenSanctions (nomenklatura/yente)**
- logic-v1 feature weights:
  - `name_literal_match` 1.00 (exact).
  - `name_fingerprint_levenshtein` 0.90: Damerau-Levenshtein on org-name fingerprints, where the fingerprint simplifies entity-type words, e.g. "Limited" to "Ltd".
  - `person_name_jaro_winkler` 0.80.
  - `identifier_match` 0.85.
  - Penalty `orgid_disjoint` −0.20 (different registration numbers).
  - [OpenSanctions matcher docs](https://www.opensanctions.org/matcher/)
- logic-v2 feature weights:
  - `name_match` 1.00.
  - LEI/BIC 0.95, `identifier_match` 0.85.
  - `weak_alias_match` 0.80.
  - `address_prop_match` only 0.20.
  - Penalties: `country_mismatch` −0.20 and date/gender mismatches.
  - [OpenSanctions matcher docs](https://www.opensanctions.org/matcher/)
- Other matchers: `name-based` (Jaro-Winkler + Soundex, no penalties) and `regression-v1` (logistic regression that gives high scores only for multi-attribute matches). logic-v1 is now deprecated. — [OpenSanctions tuning](https://www.opensanctions.org/docs/api/tuning/)

### Inferences
- **Scores add up; they are not averaged.** In OpenSanctions a name is the base score, a matching address adds only +0.20, and conflicting evidence (country, registration ID) **subtracts**. We should add explicit negative features, e.g. a different house number, a different postcode, or a different distinctive name token.
- **Shared addresses need a second signal.** Senzing's MULT_FF rule and its "same address, different name gives only possibly related" example both say an address match needs another corroborating attribute or a close name before merging. This handles office buildings and malls; Meta's "Gap Outlet vs Sears Outlet" is the same failure.
- **Use Yelp's word-alignment idea as a feature.** Compute the IDF-weighted token overlap of names, and separately the most discriminative *non*-shared token. Mismatched high-IDF tokens are strong negatives.
- **Weight distinctiveness in two places.** Use Splink's TF adjustment and Senzing's generic detection to (a) downweight common name tokens (chain brands, "restaurant", "kirana", "pharmacie") and (b) turn off addresses or phones shared by more than N records as match evidence.

### Gaps
- Google Knowledge Graph / Maps place dedup, LinkedIn/Microsoft/Bing company matching, Uber and Airbnb POI dedup: no primary sources were retrieved this session.
- Tamr (ML mastering with active learning) and Zingg (learned blocking plus active learning): docs were not fetched (the Zingg docs redirect failed).
- Dun & Bradstreet (proprietary match grades/confidence codes) and OpenCorporates: not verified.

## Q3. Common patterns: blocking, clustering safeguards, entity-centric resolution, generic names, legal form

### Takeaway
The systems converge on the same design:
- Several strict blocking keys (or a search index) that favor recall.
- A precision-oriented pairwise scorer that weights tokens by distinctiveness and includes explicit negative evidence.
- Clustering that does not trust raw transitive closure.
- Frequency-based suppression of generic values (shared addresses, chain names).

### Cited Findings
- Multi-pass or OR'd strict blocking rules. — [Splink](https://moj-analytical-services.github.io/splink/topic_guides/blocking/blocking_rules.html)
- A low-threshold search index for recall, then a strict classifier. — [Yelp](https://engineeringblog.yelp.com/2015/05/seeing-double-on-yelp.html)
- Geo cell plus TF-IDF candidate generation. — [Foursquare](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/); [dbreunig H3](https://www.dbreunig.com/2024/09/27/conflating-overture-points-of-interests-with-duckdb-ollama-and-more.html)
- Connected components is standard. — [Splink clustering](https://moj-analytical-services.github.io/splink/api_docs/clustering.html)
- Cluster-level error detection exists because pairwise errors hide in clusters. — [Amazon GNN IC detection](https://arxiv.org/abs/2105.05957)
- Entity-centric scoring (a record against the candidate entity's accumulated attributes) and re-evaluation when a value turns generic. — [Senzing PDF](https://senzing.com/wp-content/uploads/Principle-Based-Entity-Resolution-092519.pdf)
- Legal-form handling: OpenSanctions normalizes legal-form words ("Limited" to "Ltd") inside the name fingerprint rather than treating them as a separate feature. — [OpenSanctions matcher](https://www.opensanctions.org/matcher/)
- Senzing treats "ABC Company" vs "ABC Enterprises Inc." at the same address as only a *possible* match. — [Senzing PDF](https://senzing.com/wp-content/uploads/Principle-Based-Entity-Resolution-092519.pdf)
- Canonical record selection after clustering ("best page"). — [Meta patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10282434)
- Multipartite (multi-source, one-to-one) ER can be formulated as a multidimensional assignment problem. Greedy plus very-large-scale neighborhood search beats plain greedy. The paper is academic (Kammerdiner et al.) with no company affiliation. — [arXiv 2112.03346](https://arxiv.org/abs/2112.03346)

### Inferences (actionable for AMLC, simplified)
1. **Blocking**: union of strict keys: (normalized postcode + first distinctive name token), (city + name-token MinHash/char-3gram TF-IDF top-k), (phone if present), (house number + street token). Measure candidate recall on train for each key, as Splink recommends.
2. **Pairwise scorer**: a LightGBM/RF classifier on:
   - Name: Jaro-Winkler, token-set ratio, IDF-weighted Jaccard, and the max IDF among unshared tokens (Yelp alignment idea).
   - Name match after stripping legal forms (Pvt Ltd/LLP/SARL/SAS/EURL/Inc/LLC) and generic category words.
   - Address: house-number equality or conflict, postcode equality, street-token similarity.
   - Frequency of the shared address and shared name across the dataset (Senzing/Splink).
   - Script flag and transliterated-name similarity.
3. **Our one-to-one structure** (EDA: each S2/S3 record maps to at most one S1). Enforce it as a constraint: per S2/S3 record keep only the best-scoring S1 candidate above the threshold (mutual-best / Hungarian per block). This is a cheap version of the multipartite assignment formulation and it strongly limits over-merging, since no chains can form.
4. **Clustering guard**: anchor clusters on S1, attaching S2/S3 records to S1 anchors (entity-centric, as in Senzing), rather than taking transitive closure over all edges. Reject a cluster if its S2 and S3 members disagree strongly (the Amazon IC-detection idea, done with a rule).
5. **Precision control**: pick the threshold on a validation set to maximize F0.5 directly. Yelp's F0.1 table shows that raising precision from 96.6% to 99.1% while slightly increasing recall came from better *negative* features, not from a looser threshold. Use a two-tier threshold (AWS/Overture): auto-accept high scores, and give grey-zone pairs extra checks (e.g. an LLM ≤8B judge on a small subset if CPU time allows) or reject them.
6. **Rule cascade first**: exact normalized name + same postcode, then fuzzy rules, then the model or embeddings on the residue. The dbreunig cascade shows each stage adding matches cheaply. Embeddings are slow on CPU, so use them only on the residue.

### Gaps
- No source gave a concrete algorithm for "bridge-edge" cluster splitting in a production business ER system. The only hit on bridge edges was a low-quality GitHub issue, which I excluded as evidence.
- Transliteration handling (Indic scripts to Latin) in industrial business matching was not documented in any source found.

## Q4. Reported precision/recall and the false-merge vs miss trade-off

### Takeaway
Published operating points in production business dedup are strongly precision-first: Yelp auto-merges at 99.1% precision and 27.7% recall, and routes lower-confidence pairs to human review. Our metric (F0.5) is less extreme, so our operating point should sit at higher recall than Yelp's while keeping precision well above recall.

### Cited Findings
- Yelp: F0.1 = 0.966 (P 99.1%, R 27.7%); exact-only baseline P 97.1%, R 13.4%. — [Yelp](https://engineeringblog.yelp.com/2015/05/seeing-double-on-yelp.html)
- Yelp Business Match (a retrieval/matching task, not auto-merge): F1 rose from 91% to 95% with learning-to-rank. — [Yelp LTR](https://engineeringblog.yelp.com/amp/2014/12/learning-to-rank-for-business-matching.html)
- Overture uses two-model agreement for labels, sending disagreements to humans. — [Overture](https://docs.overturemaps.org/blog/2025/05/21/release-notes/)
- AWS uses split thresholds: high for automated merges, lower for inclusion. — [AWS 2026-09](https://aws.amazon.com/about-aws/whats-new/2026/09/entity-resolution-record-confidence/)
- Senzing uses three outcome tiers (match / possible match / possibly related) so that ambiguous cases are not merged. — [Senzing PDF](https://senzing.com/wp-content/uploads/Principle-Based-Entity-Resolution-092519.pdf)

### Inferences
- F0.5 at a given (P, R): P 0.95 / R 0.60 gives about 0.85, and P 0.85 / R 0.85 gives 0.85. The two are roughly equal, so recall still matters. Tune the threshold empirically rather than copying Yelp's extreme precision.
- "Smaller candidate sets rewarded" means we should output only the top match (or none) per record. This fits the industrial "possible match is not merged" tiering.

### Gaps
- No published P/R figures were found for Google, Meta Places, Foursquare's production resolve model, Senzing, Tamr or AWS ER on business data.

## Q5. Business-name normalization (legal suffixes, abbreviations, transliteration)

### Takeaway
The public detail is thin. OpenSanctions normalizes legal forms inside a name fingerprint before a Levenshtein comparison. Yelp uses synonym-aware field matching and detects person-named businesses. AWS supports Soundex/phonetic thresholds. No industrial source documented Indic transliteration.

### Cited Findings
- OpenSanctions fingerprinting simplifies entity-type names ("Limited" to "Ltd") and compares the fingerprints with Damerau-Levenshtein (weight 0.90). — [OpenSanctions matcher](https://www.opensanctions.org/matcher/)
- Yelp uses synonym-aware matching on fields, an NER model for person-named businesses, and a word-alignment classifier on names. — [Yelp](https://engineeringblog.yelp.com/2015/05/seeing-double-on-yelp.html)
- AWS ER fuzzy rules support Soundex, Levenshtein and cosine. — [AWS ER](https://docs.aws.amazon.com/entityresolution/latest/userguide/rule-based-mw-advanced.html)
- Foursquare's resolve model is trained on multilingual, multi-country labeled data, i.e. it learns across languages rather than relying on rules. — [Foursquare](https://foursquare.com/resources/blog/products/technical-guide-to-foursquare-places-part-2-how-does-foursquare-get-location-data-right/)

### Inferences
- Build a legal-form dictionary by country:
  - US: Inc, LLC, Corp, Co, Ltd.
  - India: Pvt Ltd, Private Limited, LLP, OPC, "& Sons", "Enterprises", "Traders".
  - France: SARL, SAS, SASU, EURL, SA, SNC, "Ets"/"Établissements".
- Map each legal form to a canonical token. Keep it as a separate low-weight feature (legal-form conflict is a mild negative) and exclude it from the core-name comparison, following OpenSanctions/Senzing.
- For Indic scripts, transliterate to Latin with a local, rule-based library (e.g. indic-transliteration, or unidecode as a fallback), then compare phonetic keys. Industry uses multilingual training data, which we can approximate with a multilingual embedding model ≤8B only on the hard residue.

### Gaps
- The open-source `rigour` library (OpenSanctions) holds org-type and name-normalization tables, but I did not fetch it this session. It is worth checking for legal-form lists (licence should be checked; I believe it is MIT, unverified).
- No source on how Google, LinkedIn or D&B handle abbreviation expansion or transliteration was retrieved.
