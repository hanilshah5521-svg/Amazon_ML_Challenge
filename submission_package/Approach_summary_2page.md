# Business Entity Resolution: approach summary (2 pages)

**Team:** [YOUR TEAM NAME] · **Task:** for each Source 1 business, find all matching Source 2/3 records (macro F0.5)

## Pipeline
`raw TSVs → normalize → per-state candidate search (top-40) → learned cut (best 8) → ~75 pair features → LightGBM stage 1 → stage 2 (group features) → per-record expected-F0.5 decision → one owner per Source 2/3 record → TSVs`

1. **Normalization.** Lower-case, accent stripping, punctuation to spaces, Indic-script words mapped to English with a dictionary learned from the training pairs (1,347 words) and
   anyascii as fallback, legal forms and stop words removed (`name_core`, `name_key`), USPS/AFNOR street types, state canonicalization, alias markers (f/k/a, dba), look-alike digits.
2. **Candidate generation.** Per country and within the same state: character 3-gram TF-IDF on the name (spaces removed) and word TF-IDF on the address, top-30 each, plus a pass against
   pool records without a detectable state (a third of French Source 2/3 records). The union is ranked by `name_sim + addr_sim` and the top 40 are stored.
3. **Learned cut (candidate-set size).** A small LightGBM on 28 cheap pair features (similarities, ranks, gaps to the best candidate, state relation, house-number overlap, token
   Jaccards) re-ranks the 40 and keeps the **best 8 per Source 1 record**. Out-of-fold recall of true matches (200k held-out records): India 0.971, US 0.981 with 8 candidates, against
   0.960 / 0.977 for the raw top-15. `candidate_pairs.tsv` is exactly this set (13.9M pairs for 1.73M records).
4. **Matcher.** Stage 1 (LightGBM, ~75 features): string similarities (Jaro-Winkler, token sort/set, partial), legal/entity-type changes, compound house numbers (exact, subset, one part
   changed, truncation), IDF-weighted shared and unmatched distinctive words, invented trade name, acronym, website name, state agree/conflict/missing, name and address crowding, and
   **decoy / noise word** indicators. Stage 2 adds group features from the stage-1 probabilities of the record's other candidates. 5-fold CV grouped by Source 1 record, 400k training records.
5. **Decision.** Per record, accept the top-n candidates that maximize the expected F0.5 (n may be 0), or a probability threshold; picked on half A of the held-out records and reported on
   half B; then each Source 2/3 record goes only to its highest-scoring Source 1 record.

## What is new in this version
- **Learned cut instead of the raw top-15:** more recall with about half the candidates, which also halves the matcher's cost and allowed twice the training data.
- **Decoy vs noise words:** a generic word added by the noise is a *decoy* if the house numbers then disagree (partners, holdings, group, enterprises, groupe, holding, participations …) and *noise* if they agree
  (center, service, services …). The per-country lists are learned without labels from candidate pairs (train for India/US, the unlabelled test candidates for France).
- **Open-set countries:** no country feature; countries are taken from the data; countries without labels use the strictest labelled-country decision settings. Country-specific handling exists
  only in the text normalization.

## Results (held-out half B, macro F0.5)
| Step | India | US | Candidates per record |
|---|---|---|---|
| previous pipeline (0.967 on the leaderboard) | 0.9751 | 0.9799 | 15 |
| + learned cut | 0.9772 | 0.9813 | 8 |
| + decoy/noise words | 0.9781 | 0.9814 | 8 |
| + 400k training records | 0.9790 | 0.9818 | 8 |
| **country-agnostic (final)** | **0.9788** | **0.9818** | 8 |

Tested and dropped (no gain or worse): early-stopped boosting rounds, extra normalization rules with S1-vocabulary spell correction (India −0.0015), 10 instead of 8 candidates.
Test output: 5,736,957 matches (3.31 per record), 5.8% of Source 1 records left empty (training truth 5.6%).

## Error analysis
False accepts and no-match errors cost only ~0.004: planted decoys with the same name and street but a different house number or legal form. Most remaining loss is an information limit:
candidates with no address whose name is shared by many Source 1 records (a chain name with 4+ Source 1 records has a 4.3% true rate and the model's probabilities are calibrated there), plus true matches that
blocking never retrieves (India 0.0086, US 0.0058). France has no labels and cannot be measured locally.

## Compliance
No external data, APIs, geocoding or internet lookups. The only model is LightGBM (MIT) trained from scratch on the provided training data; no pretrained models. Statistics from the unlabelled test
files (IDF, vocabulary, the France decoy-word list) are unsupervised. Reproduce with `bash code/business_entity_resolution/src/run_final.sh` (about 2 h 15 min, 16 GB RAM); dependencies are pinned.
