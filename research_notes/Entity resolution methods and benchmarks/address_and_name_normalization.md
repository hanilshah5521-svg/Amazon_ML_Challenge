# Address parsing/normalization and multilingual business-name normalization (US, India, France)

Context: AMLC 2026 entity resolution. We need CPU-only tools (16 GB RAM, about 48h), MIT/Apache models of 8B params or less, and **no external data lookup, APIs or geocoding**. Every tool below is tagged with **License / Size / CPU / External-data risk**.

## Q1. Address parsers and normalizers (libpostal, usaddress, deepparse, Indian parsers, French/US abbreviation norms)

### Takeaway
libpostal is MIT and fast, but its ~1.8 GB model was trained on OpenStreetMap and OpenAddresses, and it is hard to build on Windows. That makes it the riskiest "bundled external data" case, so avoid it or ask the organisers first. Better options are usaddress (MIT, parse only, no normalization) for US tokenization and hand-written abbreviation tables (USPS Pub 28, AFNOR list) inside our existing rule-based normalizer. Deepparse is LGPLv3, not MIT/Apache, and only reaches ~64–70% zero-shot on India.

### Cited Findings
- **libpostal**: MIT license. Trained with a CRF on "over 1 billion addresses" from OpenStreetMap and OpenAddresses. Default model is ~1.8 GB on disk (Senzing model ~2.2 GB) and uses under 1 GB RAM per process. Speed is 10–30k addresses/sec on a single thread, with 99.45% full-parse accuracy on held-out data. It covers abbreviation expansion and transliteration in 60+ languages. Windows needs an MSys2/MinGW build — [libpostal README](https://github.com/openvenues/libpostal)
- **usaddress**: MIT license. A CRF parser (python-crfsuite) that outputs labels such as AddressNumber, StreetName, StreetNamePostType, OccupancyType, OccupancyIdentifier, PlaceName and StateName. `parse()` returns token/label tuples. `tag()` merges consecutive components and adds an address type. It "cannot normalize"; normalization lives in the separate usaddress-scourgify package — [usaddress README](https://github.com/datamade/usaddress)
- **deepparse**: **LGPLv3** license. Pretrained FastText and BPEmb seq2seq models, hosted on HF, trained on 20 countries that include France and the US but not India. Accuracy is ~99.6% for France (FastText) and **70.3% (FastText) / 63.68% (BPEmb) zero-shot for India** — [deepparse README](https://github.com/GRAAL-Research/deepparse); [deepparse paper](https://arxiv.org/pdf/2311.11846)
- **indian-address-parser** (community project): MIT license. IndicBERTv2 plus a CRF that extracts 15 entity types (house no., floor, block, sector, gali, colony, area, subarea, khasra, pincode, city, state…). Reports 94.3% F1 on 150 test samples and ~25 ms/address on CPU. It handles Hindi and English and **includes fuzzy matching against a Delhi gazetteer** (external data) — [GitHub](https://github.com/howdoiusekeyboard/indian-address-parser)
- Indian address research exists (Word2Vec+TF-IDF, Bi-LSTM and BERT for sub-region classification of North/South Indian city addresses) — [arXiv 2007.03020](https://arxiv.org/pdf/2007.03020)
- **USPS Publication 28, Appendix C1**: about 235–240 primary street suffixes, each listing common variants and one standard abbreviation. Examples: STREET→ST (variants STRT, STR); AVENUE→AVE (AV, AVEN, AVENU, AVN, AVNUE); ROAD→RD; BOULEVARD→BLVD (BOUL); DRIVE→DR (DRIV, DRV); CIRCLE→CIR (CIRC, CIRCL, CRCL, CRCLE) — [USPS Pub 28 C1](https://pe.usps.com/text/pub28/28apc_002.htm)
- **French AFNOR XP/NF Z10-011 authorized street-type abbreviations**: ALL=Allée, AV=Avenue, BD=Boulevard, CTRE=Centre, CCAL=Centre commercial, IMM=Immeuble, IMP=Impasse, LD=Lieu-dit, LOT=Lotissement, PAS=Passage, PL=Place, RES=Résidence, RPT=Rond-point, RTE=Route, SQ=Square, VLGE=Village, ZA, ZAC, ZAD, ZI. Types not on the list are cut to their first 4 letters (CHEMIN→CHEM, SENTIER→SENT) — [SITI LR summary of the AFNOR norm](http://ressources.sitilr.fr/document/types-de-voie-norme-afnor-xp-z-10-011/)
- The full NF Z10-011 standard is a paid AFNOR document. It recommends abbreviating street types first when space is short — [AFNOR boutique](https://www.boutique.afnor.org/en-gb/standard/nf-z10011/mailing-address-how-to-write-a-mailing-address-rules-for-the-presentation-o/fa178533/40541); [comprendrelatypo wiki](https://comprendrelatypo.fandom.com/fr/wiki/Abr%C3%A9viations_postales)

### Inferences
- **External-data judgment**:
  - Hand-coding the USPS and AFNOR tables into our rules is a standard, low-risk step: it is domain knowledge, not a lookup database.
  - libpostal's model is literally an OpenStreetMap-derived artifact. It also has place/toponym dictionaries, so it effectively does "offline geocoding-lite". That makes it the grey-area case, so ask the organisers before using it.
  - The indian-address-parser's Delhi gazetteer is clearly external data.
- For France, which appears only in test so nothing can be learned from labels, a rule table is the main tool:
  - Expand the AFNOR codes plus the 4-letter truncations (CHEM, SENT…).
  - Add informal forms: R./R→Rue, BLD/BVD→Boulevard, CHE/CH→Chemin, FG→Faubourg, QU/QUAI.
  - Map "N°"/"no" to nothing before house numbers, and attach bis/ter/quater as a house-number suffix.
  - "Rue" is not on the official abbreviation list (it is already short), so "R." is an informal variant that needs its own rule.
- For region vs department (Nord vs Hauts-de-France) and US hamlet vs town (Brentwood vs Islip), a static mapping table would amount to an external gazetteer. Safer: learn co-occurrence from the data itself (test addresses of the same entity across sources), or drop admin-level tokens from the similarity and rely on the street, number and postcode.
- Deepparse is excluded on license grounds (LGPL, not MIT/Apache), and its India accuracy is poor anyway.

### Gaps
- I found no authoritative source on how US hamlets/CDPs relate to towns (e.g. Brentwood is a hamlet inside the Town of Islip) that would be usable without a gazetteer.
- I did not verify the unofficial French variants (R., CHE, FG) against La Poste's own documentation. The La Poste PDF ([presse-poste guide](https://lastation.laposte.fr/sites/p8_u1/files/2023-03/SP8855-Volume%202_V1.11%20-%20Adressage%20des%20plis.pdf)) was found but not read.
- I found no Shipsy/Delhivery/Flipkart public papers with reusable Indian abbreviation lists.

## Q2. Indic→Latin transliteration tools

### Takeaway
Our learned Indic→English word dictionary (96% coverage) is probably the best primary tool, because it produces English spellings ("traders"), not phonetic romanizations. For the remaining ~4% out-of-vocabulary tokens:
- **indic_transliteration (sanscript)**: MIT, pure Python, instant. It gives a rule-based romanization that can then be compared phonetically.
- **IndicXlit**: MIT, ~11M params, has an Indic→Roman model. It is a neural fallback but installs painfully via fairseq on Windows.
- **Aksharamukha**: AGPL-3.0, so avoid it.

### Cited Findings
- **IndicXlit**: MIT license. Transformer with ~11M params (6 encoder + 6 decoder layers, 256-dim embeddings, 4 heads, 1024 FFN). Covers 21 Indic languages, including Bengali, Gujarati, Hindi, Kannada, Malayalam, Oriya, Panjabi, Tamil, Telugu and Urdu. Offers both Roman→Indic and **Indic→Roman (v1.0)** models. Needs fairseq installed from source. Example reported top-1 accuracy: Hindi 60.56 (En→Indic, Dakshina) — [IndicXlit README](https://github.com/AI4Bharat/IndicXlit)
- `pip install ai4bharat-transliteration` fails on Windows with "Failed building wheel for fairseq". The workaround is to install indic-nlp-library, then fairseq from a git clone with `pip install --editable ./` — [IndicXlit issue #21](https://github.com/AI4Bharat/IndicXlit/issues/21); [libraries.io](https://libraries.io/pypi/ai4bharat-transliteration)
- **indic_transliteration (sanscript)**: MIT license. Covers Devanagari, Telugu, Tamil, Kannada, Bengali, Gujarati, Gurmukhi, Malayalam and Oriya. Schemes include ITRANS, IAST, HK, OPTITRANS and Velthuis. Has a `to_lay_indian()` simplification ("kRShNa"→"krishna"). `pip install indic_transliteration` — [GitHub](https://github.com/indic-transliteration/indic_transliteration_py)
- **Aksharamukha**: **GNU AGPL 3.0**. Covers 120 scripts and 21 romanizations and handles orthographic conventions — [GitHub](https://github.com/virtualvinodh/aksharamukha)
- An LLM-vs-specialist transliteration benchmark for Indian languages exists — [arXiv 2505.19851](https://arxiv.org/pdf/2505.19851)

### Inferences
- Rule-based romanization of "ट्रेडर्स" gives roughly "treDarsa"/"tredars", not "traders". Two follow-ups make it matchable:
  - Normalize it: lowercase, strip the trailing inherent "a", collapse retroflex/aspirate distinctions.
  - Then compare it phonetically (Q3) or with char n-grams against the English side.
- Adding IndicXlit (small, 11M params, CPU-friendly) is probably worth it only if the OOV tokens actually hurt F0.5. Its model weights are MIT-licensed code/artifacts, not a lookup database.
- Plain script-to-script tools (sanscript) carry no external-data risk; they are character tables.

### Gaps
- I found no published accuracy for IndicXlit **Indic→Roman on English loanwords** (e.g. ट्रेडर्स→traders).
- I found no measured CPU throughput for IndicXlit.
- I did not check polyglot's or the IndicNLP library's transliteration licenses in this pass.

## Q3. Phonetic encodings and string similarity for noisy / cross-script names

### Takeaway
Use **jellyfish** (MIT, Rust core, fast): Soundex, Metaphone, NYSIIS, Match Rating codex, plus Jaro-Winkler and Damerau-Levenshtein. Avoid **abydos**: it has Beider-Morse and Double Metaphone but is GPL-3.

### Cited Findings
- **jellyfish**: MIT license, Rust core with Python bindings.
  - String comparisons: Levenshtein, Damerau-Levenshtein, Jaccard, Jaro, Jaro-Winkler, Hamming, Match Rating comparison.
  - Phonetic encodings: American Soundex, Metaphone, NYSIIS, Match Rating codex.
  - Source: [jellyfish GitHub](https://github.com/jamesturk/jellyfish)
- **abydos**: **GPL-3** license. The authors say GPL3 is "the only license compatible" with the ported parts, e.g. Beider-Morse, which was ported from GPL3 PHP. It includes Soundex, Double Metaphone, NYSIIS, Alpha-SIS, BMPM, Jaro-Winkler and skeleton/omission keys — [abydos FAQ](https://abydos.readthedocs.io/en/latest/faq.html); [GitHub](https://github.com/chrislit/abydos)

### Inferences
- Pipeline for Indic tokens not in our dictionary: sanscript romanization → simplification → jellyfish NYSIIS/Metaphone key. Use that key as a blocking key and as a feature against English tokens.
- Soundex/Metaphone are tuned for English. For French names, a lightweight custom rule set (drop final silent consonants, eau/au/o→o, ph→f, qu→k) may beat them. This is untested.
- The GPL-3 license rules out abydos, since only MIT/Apache is allowed.

### Gaps
- I did not confirm the license of a standalone Double Metaphone package; jellyfish does not include Double Metaphone.
- I found no Indian-specific phonetic algorithm with a verified license.

## Q4. Company-name normalization (legal forms, generic terms, dba/fka)

### Takeaway
**cleanco** (MIT) strips legal suffixes and is a good base. Its France list is decent, but its India list is very thin ("ltd.", "pvt. ltd.", "psu", "pse"), so extend it by hand.

### Cited Findings
- cleanco is MIT-licensed. `basename("Some Big Pharma, LLC")` returns "Some Big Pharma", and `typesources()`/`countrysources()` with `matches()` return the entity type and candidate countries — [cleanco README](https://github.com/psolin/cleanco)
- cleanco term lists — [termdata.py](https://raw.githubusercontent.com/psolin/cleanco/master/cleanco/termdata.py):
  - France: ei, eurl, fcp, gie, sarl, sas, sasu, sca, sci, scop, scs, sem, sep, sicav, snc, "societe civile immobiliere", sogepa.
  - India: ltd., pse, psu, pvt. ltd.
  - US: & co, co, company, corp, corporation, inc, incorporated, l.l.p., limited, llc, llp, lp, ltd, pllc.

### Inferences
- Add India forms by hand, then normalize punctuation before running cleanco:
  - Private Limited, Pvt, Pvt., P. Ltd, (P) Ltd, Pvt Ltd, Private Ltd
  - LLP, OPC, One Person Company
  - "& Sons", "& Co"
  - Native-script forms such as प्राइवेट लिमिटेड; these map through our dictionary first.
- For France, also add SA and "et Cie". Treat "Ets"/"Établissements" as generic prefixes.
- Store the legal form as a separate feature instead of throwing it away: SAS vs SARL with the same base name is weak evidence against a match.
- Down-weight generic trade words (traders, enterprises, stores, restaurant, boulangerie, pharmacie) with IDF learned from our own corpus. This avoids needing an external stopword list.
- dba/fka: split on "dba", "d/b/a", "doing business as", "fka", "aka", "t/a", "trading as". Keep both parts as alternative names and take the max similarity.

### Gaps
- I found no source measuring cleanco's accuracy.
- I found no standard reference list for website/handle-name normalization; derive one from the data (strip www., .com/.in/.fr, @, split camelCase).

## Q5. Multilingual embedding models (MIT/Apache, CPU) for name similarity

### Takeaway
multilingual-e5-small (MIT, 118M, 384-d) and paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M, 384-d) are license-compliant and CPU-feasible. LaBSE (Apache-2.0) is larger (~0.5B, 768-d). The evidence is weak that sentence encoders handle **cross-script names** well, so embed after transliteration/dictionary mapping, not on raw Indic script.

### Cited Findings
- **multilingual-e5-small**: MIT license, 118M params, 384-dim, ~100 languages. Trained with "query: "/"passage: " prefixes, and omitting them costs quality — [Superlinked E5 guide](https://superlinked.com/blog/e5-embedding-family-guide); [HF model card](https://huggingface.co/intfloat/multilingual-e5-small)
- **paraphrase-multilingual-MiniLM-L12-v2**: Apache-2.0 license, 118M params, 50+ languages, 384-dim — [PromptLayer model page](https://www.promptlayer.com/models/paraphrase-multilingual-minilm-l12-v2/); [SBERT pretrained models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- **LaBSE**: Apache-2.0 license, ~0.5B params, 109 languages, 768-dim — [HF LaBSE](https://huggingface.co/sentence-transformers/LaBSE). The SBERT docs note LaBSE is weaker for similarity of pairs that are not translations — [SBERT docs](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- **Cross-script name evidence** (informal gist): paraphrase-multilingual-MiniLM gave the *same* name across scripts low cosine scores — Igor Berezovsky vs Игорь Березовский 0.323, Oleg Konovalov vs Олег Коновалов 0.339 — which is lower than *different* Latin names (Sam Smith vs Tom Jones 0.456). Latin variants scored high (Russell H Jurney vs Russell Jurney 0.952) — [rjurney gist](https://gist.github.com/rjurney/35e18a749bd3064ad1c57ea4420fb01c)

### Inferences
- Do not rely on raw cross-script embeddings. Map to English with the dictionary first, then embed. The embedding mainly helps with abbreviations and synonyms in Latin script.
- Throughput guess (not measured): a 118M model runs at maybe ~200–1000 short strings/sec on a laptop CPU with batching, i.e. roughly 20–80 min per 1M names. ONNX/int8 quantization could speed this up about 2x. Benchmark on our own machine before committing.
- Model weights (MIT/Apache) are generally treated as models, not "external databases", under competition rules. Still confirm that pretrained models are allowed, given the ≤8B limit implies they are.

### Gaps
- I found no measured CPU speed per 1M strings for these models.
- I found no benchmark of IndicSBERT/MuRIL on transliterated business names; I did not verify their licenses in this pass.

## Q6. US address normalization: USPS Pub 28 suffixes/units; hamlets vs towns

### Takeaway
USPS Pub 28 Appendix C1 (street suffixes) is the canonical abbreviation table: ~235–240 suffixes, each with variants and a standard abbreviation. Hard-code it, plus the unit designators (C2) and directionals. Hamlet/CDP vs town cannot be resolved offline without a gazetteer, so treat the city as a soft feature.

### Cited Findings
- Pub 28 C1 has three columns: primary name, common variants, standard abbreviation, e.g. STREET/STR/STRT→ST, AVENUE/AV/AVEN/AVENU/AVN/AVNUE→AVE, BOULEVARD/BOUL→BLVD — [USPS Pub 28 C1](https://pe.usps.com/text/pub28/28apc_002.htm)
- usaddress tags OccupancyType/OccupancyIdentifier, so it can split out units (Suite, Apt) for separate normalization — [usaddress](https://github.com/datamade/usaddress)

### Inferences
- Copying the Pub 28 tables into code is normalization knowledge (a public standard), not a data lookup, so risk is low.
- US state code↔name (50 + DC) is a trivial static table; comparable in risk to the abbreviation lists.
- For hamlet vs town, key matches on street number + normalized street + ZIP, and down-weight disagreement on the city token when the ZIP matches.

### Gaps
- I did not fetch the Pub 28 unit-designator table (Appendix C2) in this pass.
- I found no source on hamlet/CDP handling in entity resolution.
