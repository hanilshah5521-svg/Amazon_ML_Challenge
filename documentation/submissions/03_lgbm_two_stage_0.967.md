# Submission 3: two-stage LightGBM with v2 features (leaderboard 0.967)

- **Submitted:** 26 Sep 2026, ~10:00 IST
- **Leaderboard (public):** **0.967** (up from 0.915; current best)
- **Output:** `output/v3_lgbm_two_stage/matching_results.tsv`, SHA-256 prefix `b5edd37b8508fe00`
- **Reproduce:** git branch `0.967`, then `bash code/business_entity_resolution/src/run_v2.sh`
  (about 2.5 h on a 16 GB / 18-thread laptop; versions pinned in `requirements.txt`)

## What changed since submission 2 (0.915)
| Change | Evidence it was needed |
|---|---|
| **Normalizer v2**: French rules (N°, apartment/floor drops, French generic words, all/q/passage fixes), AFNOR street types, street-type typo fix (aveue→avenue), `12bis` split, 5/8/6 look-alikes (5ARL, 8ordeaux, 6irard); **anyascii** replaces GPL unidecode | France token-shift mining; error review |
| **Per-state candidate search**: top-30 by name + top-30 by address within the state, no common-word dropping, plus a pass for records with no state; top-15 go to the model | True matches share the state 100% of the time; candidate-search misses were the largest loss |
| **~45 new features** (`features_v2.py`): entity-type added/dropped/changed and Private added/dropped; compound house numbers kept whole with the kind of difference (exact, one part changed, truncation, ±3, range); rarity-weighted distinctive vs generic unmatched words, first word, invented trade name, acronym, website name; state agree/conflict/missing; name and address crowding; same-name candidates with the best address | Error analysis of the 0.915 model: every false accept was a planted decoy |
| **Stage 2 (group model)**: stage-1 probabilities of the record's other candidates, whether strong candidates come from both sources, second-best and gap, the top candidate's warning flags, deviation from the group majority | No-match analysis: decoys are lone candidates with 10× more number conflicts |
| **Per-record expected-F0.5 decision** (can predict nothing), chosen against a plain threshold on half A and scored on half B | Protects records with no true match |
| **200k training records** (was 20k): 3M labelled pairs | Learning curve: +0.003 per doubling |

## Measured results (held-out half B of 200k training records)
| Model | India F0.5 | US F0.5 |
|---|---|---|
| Base 24 features (the 0.915 feature set) with the new candidates and more data | 0.937 | 0.947 |
| + v2 features (stage 1) | 0.975 | 0.979 |
| **+ stage 2 + expected-F0.5 decision** | **0.975** | **0.980** |

- **Candidate recall at top-10:** India 90.5% → 95.6%, US 95.0% → 97.4% (98.0% at top-15).
- **Decision:** expected-F0.5 for all countries. France uses the stricter (US) settings, because
  France has no labels.
- **Test output:** 5,733,668 matches; 5.8% of records left empty (training no-match rate 5.6%).
  Matches per record: France 3.28, India 3.27, US 3.37 (training truth 3.46).
- **Leaderboard 0.967** vs held-out ~0.977 implies France ≈ 0.93 (an inference, not a
  measurement).
- Validator with `--check-ids`: PASS.

## Incidents
- The overnight prediction step ran out of RAM (all raw text reloaded per chunk, 12 workers) and
  hung for ~5.5 h without failing. The fix: load only each chunk's records, cache the country
  tables, score each chunk immediately, use 8 workers, and add a 3-hour timeout.
- `output/` was overwritten. Outputs are now versioned (`output/v<N>_<tag>/`, never
  overwritten).

## Known limitations (for submission 4)
- France is still unmeasured. It uses the US settings and has no French synthetic validation.
- Candidate file: 15 per record (~1.73M rows; 357 MB). Not yet reduced for the final ranking.
- Not yet tried: transitive Source 2↔3 expansion, reverse search, ensembles, training on all
  2.2M records.
