"""Text normalization for business names and addresses.

Indic-script words are mapped to English using a word dictionary learned from the
training ground truth (Indic names are word-by-word transliterations of the S1 name),
falling back to unidecode for unseen words. Everything is then lowercased, punctuation
is turned into spaces, and abbreviations / legal suffixes / state names are
canonicalized per country.

Usage:  python normalize.py build-dict      # learn Indic->English dictionary from train
        python normalize.py <train|test>    # write work/<split>_norm.parquet
"""
import json
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from multiprocessing import Pool

import polars as pl
from rapidfuzz.distance import OSA
from unidecode import unidecode

from common import WORK, read_pairs, read_source

DICT_PATH = WORK / "indic_dict.json"
INDIC = re.compile(r"[ऀ-෿]")
SPLIT = re.compile(r"[\s,.\-()/&'\"!?:;\[\]<>*_+#|@{}=~`^%$\\]+")
PUNCT = re.compile(r"[^a-z0-9 ]+")
SPACES = re.compile(r"\s+")

# ---------------------------------------------------------------- name vocab
LEGAL = {  # canonical legal-form tokens (removed from name_core)
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "limited": "ltd", "ltd": "ltd",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc",
    "pc": "pc", "plc": "plc", "opc": "opc",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "societe": "ste", "ste": "ste", "cie": "co", "compagnie": "co",
    "ei": "ei", "ets": "ets", "etablissements": "ets", "etablissement": "ets",
}
# Honorifics / markers that the noise adds or drops ("dr", "sree" dropped 38%, "formerly" 100% added).
NAME_STOP = {"the", "mr", "mrs", "ms", "m", "and", "of", "dba", "fka", "aka", "et", "de",
             "du", "des", "la", "le", "les", "www", "com", "net", "org", "in", "fr", "biz",
             "france", "sri", "shri", "shree", "sree", "smt", "messrs", "dr", "formerly",
             "known", "as", "id"}
NAME_VARIANTS = {"laxmi": "lakshmi", "jay": "jai", "ganpati": "ganapati", "vinayaka": "vinayak",
                 "frs": "freres"}
# Generic business words: swapped/added as noise ("center" is added in 72% of its match
# occurrences, "care"->"center", "partners"->"services"). Dropped from `name_key` only.
GENERIC = {"center", "centre", "services", "service", "group", "holdings", "companies",
           "associates", "partners", "solutions", "enterprises", "ventures", "trading",
           "traders", "industries", "international", "management", "consultants", "consulting",
           "clinic", "care", "institute", "foundation", "trust", "society", "association",
           "technologies", "technology", "tech", "systems", "global", "brothers", "investments",
           "agency", "agencies", "network", "labs", "laboratories", "studios", "studio",
           "innovations", "brands", "advisors", "academy", "council", "league", "hospital",
           "church", "college", "dynamics", "llc", "inc", "pvt", "ltd", "corp", "co",
           # France: added by the noise in S2/S3 (e.g. "participations" 2.4% of S2/S3 vs ~0% S1)
           "participations", "holding", "distribution", "associes", "developpement", "groupe",
           "fils", "freres"}
# Alias markers: the text AFTER the marker is the entity's own name
# ("Yumavantage f/k/a Beacon Energy", "Lumonyx doing business as SJD Amis EI").
ALIAS_SPLIT = re.compile(r"\b(?:f\s*/\s*k\s*/\s*a|fka|a\s*/\s*k\s*/\s*a|aka|d\s*/\s*b\s*/\s*a|dba|"
                         r"formerly(?:\s+known\s+as)?|doing\s+business\s+as|trading\s+as)\b\s*:?",
                         re.I)
MS_PREFIX = re.compile(r"\bm\s*/\s*s\b\.?", re.I)          # "M/s Floret" -> "Floret"
# Look-alike digits inside words: "Pub1ications", "Internationa1", "1imited" (names only).
LOOKALIKE = re.compile(r"(?<=[a-z])[10](?=[a-z]|\b)|\b[10](?=[a-z]{2,})")
# 5->s and 8->b only at the start of a word followed by 3+ letters ("5arl", "8ordeaux").
LOOKALIKE_START = re.compile(r"\b([58])(?=[a-z]{3,})")
# Street-type words that the noise misspells (France: aveue/avnue/alee; US: stret/raod/drve).
STREET_TYPES = ["avenue", "allee", "boulevard", "impasse", "chemin", "route", "passage", "quai",
                "cours", "place", "street", "road", "drive", "lane", "court", "terrace"]
GLUED_SUFFIX = re.compile(r"\b(\d+)(bis|ter|quater)\b")          # 12bis -> 12 bis

# ------------------------------------------------------------- address vocab
ADDR_COMMON = {
    "road": "rd", "street": "st", "saint": "st", "avenue": "ave", "av": "ave", "drive": "dr",
    "court": "ct", "lane": "ln", "boulevard": "blvd", "bd": "blvd", "bld": "blvd",
    "highway": "hwy", "place": "pl", "circle": "cir", "trail": "trl", "parkway": "pkwy",
    "terrace": "ter", "square": "sq", "suite": "ste", "apartment": "apt", "building": "bldg",
    "floor": "fl", "north": "n", "south": "s", "east": "e", "west": "w", "number": "no",
    "near": "nr", "opposite": "opp", "post": "po", "sector": "sec", "nagar": "ngr",
    "colony": "col", "township": "twp", "township.": "twp", "mount": "mt", "fort": "ft",
    "center": "ctr", "centre": "ctr", "expressway": "expy", "freeway": "fwy", "unit": "unit",
    "door": "no", "house": "h", "hno": "h no", "chemin": "ch", "route": "rte", "allee": "all",
    "impasse": "imp", "quai": "qu", "cours": "crs",
    # USPS suffixes seen as 1:1 substitutions in true pairs (cove->cv, point->pt, ...)
    "cove": "cv", "point": "pt", "turnpike": "tpke", "crossing": "xing", "heights": "hts",
    "junction": "jct", "ridge": "rdg", "mountain": "mtn", "creek": "crk", "valley": "vly",
    "village": "vlg", "station": "sta", "springs": "spgs", "spring": "spg", "lake": "lk",
    "lakes": "lks", "estates": "est", "extension": "ext", "extn": "ext", "harbor": "hbr",
    "hills": "hls", "hill": "hl", "island": "is", "meadows": "mdws", "plaza": "plz",
    "shore": "shr", "trace": "trce", "view": "vw", "vista": "vis", "alley": "aly",
    "bend": "bnd", "bluff": "blf", "bridge": "brg", "brook": "brk", "canyon": "cyn",
    "corner": "cor", "crescent": "cres", "crest": "crst", "falls": "fls", "ferry": "fry",
    "field": "fld", "fields": "flds", "forest": "frst", "garden": "gdn", "gardens": "gdns",
    "glen": "gln", "green": "grn", "grove": "grv", "haven": "hvn", "hollow": "holw",
    "landing": "lndg", "lodge": "ldg", "manor": "mnr", "mill": "ml", "mills": "mls",
    "orchard": "orch", "pines": "pnes", "plains": "plns", "port": "prt", "ranch": "rnch",
    "river": "riv", "summit": "smt", "tunnel": "tunl", "walk": "wlk", "cross": "crs",
}
# Tokens the noise adds or drops far more often than chance (unit dropped 75%, pmb/box/cdp
# added ~100%): removed from addresses. The numbers after them are kept.
ADDR_NOISE = {"unit", "apt", "ste", "pmb", "box", "po", "county", "cdp", "twp", "borough",
              "region", "of", "null", "na"}
# India: premise-type words are swapped by the noise (plot/flat/shop -> "H", "HN").
IN_PREMISE = {"plot": "no", "flat": "no", "shop": "no", "h": "no", "hn": "no", "house": "no",
              "door": "no", "dno": "no", "no": "no", "office": "no", "bldg": "no"}
# India: old/new/alternate city names (Calcutta<->Kolkata was a top-10 learned substitution).
IN_CITY = {"calcutta": "kolkata", "bombay": "mumbai", "bengaluru": "bangalore",
           "banglore": "bangalore", "madras": "chennai", "ahmadabad": "ahmedabad",
           "vizag": "visakhapatnam", "vishakhapatnam": "visakhapatnam", "gurugram": "gurgaon",
           "trivandrum": "thiruvananthapuram", "poona": "pune", "baroda": "vadodara",
           "mysuru": "mysore", "cochin": "kochi", "calicut": "kozhikode", "banaras": "varanasi",
           "benares": "varanasi", "allahabad": "prayagraj", "belgaum": "belagavi",
           "mangalore": "mangaluru", "hubli": "hubballi", "gulbarga": "kalaburagi"}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {  # canonical code <- names/codes/variants
    # Telangana <-> Andhra Pradesh was the #1 learned address substitution (Hyderabad records
    # carry the pre-2014 state in one source), so both map to one token.
    "ap": ["andhra pradesh", "andhrapradesh", "ap", "telangana", "tg", "ts"],
    "ar": ["arunachal pradesh"],
    "as": ["assam"], "br": ["bihar", "br"], "cg": ["chhattisgarh", "chattisgarh", "cg", "ct"],
    "ga": ["goa"], "gj": ["gujarat", "gj"], "hr": ["haryana", "hr"],
    "hp": ["himachal pradesh", "hp"], "jh": ["jharkhand", "jh"], "ka": ["karnataka", "ka"],
    "kl": ["kerala", "keralam", "kl"], "mp": ["madhya pradesh", "mp"],
    "mh": ["maharashtra", "mh"], "mn": ["manipur"], "ml": ["meghalaya"], "mz": ["mizoram"],
    "nl": ["nagaland"], "od": ["odisha", "orissa", "od", "or"], "pb": ["punjab", "pb"],
    "rj": ["rajasthan", "rj"], "sk": ["sikkim"], "tn": ["tamil nadu", "tamilnadu", "tn"],
    "tr": ["tripura"],
    "up": ["uttar pradesh", "up"], "uk": ["uttarakhand", "uttaranchal", "uk"],
    "wb": ["west bengal", "westbengal", "wb"], "dl": ["delhi", "new delhi", "nct of delhi", "dl"],
    "jk": ["jammu and kashmir", "jammu kashmir", "jk"], "ch": ["chandigarh"],
    "py": ["puducherry", "pondicherry", "py"], "ld": ["ladakh"],
}
# The 17 native-script tokens found in S2/S3 addresses (co-occurrence with S1 was 100%).
INDIC_STATES = {
    "महाराष्ट्र": "maharashtra", "दिल्ली": "delhi", "उत्तर": "uttar", "प्रदेश": "pradesh",
    "ಕರ್ನಾಟಕ": "karnataka", "தமிழ்நாடு": "tamil nadu", "পশ্চিমবঙ্গ": "west bengal",
    "ગુજરાત": "gujarat", "తెలంగాణ": "telangana", "हरियाणा": "haryana",
    "राजस्थान": "rajasthan", "കേരളം": "kerala", "बिहार": "bihar", "मध्य": "madhya",
    "ఆంధ్రప్రదేశ్": "andhra pradesh", "ਪੰਜਾਬ": "punjab", "ଓଡ଼ିଶା": "odisha",
}


# State tokens are only canonicalized when a whole comma-separated address component is
# the state, so "Kansas City" or "Virginia Beach" are left alone.
STATE_COMPONENT = {
    "US": {k: f"st{v}" for k, v in US_STATES.items()} | {v: f"st{v}" for v in US_STATES.values()},
    "India": {n: f"st{c}" for c, names in IN_STATES.items() for n in names},
    # S1 uses regions, S2/S3 often use the department instead: map both to the region.
    "France": {"hauts de france": "sthdf", "nord": "sthdf", "pas de calais": "sthdf",
               "nouvelle aquitaine": "stnaq", "gironde": "stnaq",
               "pays de la loire": "stpdl", "loire atlantique": "stpdl"},
}
# France: "all"/"allee" both end as "all" via ADDR_COMMON (the old "all"->"allee" here made
# "Allée" and "All." disagree); "no" (from N°/No/#) is added in 11.7% of S2/S3 vs ~0% of S1.
ADDR_FR_DROP = {"no", "appartement", "appt", "app", "etage", "batiment", "bat", "residence",
                "res", "chez", "escalier", "esc"}
ADDR_COUNTRY = {"France": {"r": "rue", "all": "all", "q": "qu", "passage": "psg", "pass": "psg",
                           "chem": "ch", "chemin": "ch",
                           "n": "no", "pl": "pl", "fbg": "faubourg", "rte": "rte"}}
# "(India)" / "(France)" inside a name: kept in `name`, dropped from `name_core`.
PAREN_COUNTRY = re.compile(r"\(\s*(india|france|usa|u\.?s\.?a?\.?|us)\s*\)", re.I)
PAREN_MARK = "xqpc"
# Multi-letter Roman numerals only: a lone i / v / x is usually a pronoun or an initial
# ("1 800 I Care 4 U", "P V & N Acquisition").
ROMAN = {r: str(n) for n, r in enumerate(
    ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii",
     "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"]) if len(r) > 1}
_CARD = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_ORD = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
        "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
        "seventeenth", "eighteenth", "nineteenth"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
ADDR_NUMWORDS = {w: str(n) for n, w in enumerate(_CARD)} | {w: str(n) for n, w in enumerate(_ORD) if w}
ADDR_NUMWORDS |= {k: str(v) for k, v in _TENS.items()}
ADDR_NUMWORDS |= {k[:-1] + "ieth": str(v) for k, v in _TENS.items()}   # twentieth, thirtieth
_UNITS = {w: n for n, w in enumerate(_CARD[:10])} | {w: n for n, w in enumerate(_ORD[:10]) if w}
COMPOUND_NUM = re.compile(rf"\b({'|'.join(_TENS)})\s+({'|'.join(k for k in _UNITS if k != 'zero')})\b")

_DICT = None


def _indic_dict():
    global _DICT
    if _DICT is None:
        _DICT = json.loads(DICT_PATH.read_text(encoding="utf-8")) if DICT_PATH.exists() else {}
    return _DICT


def transliterate(text, mapping):
    """Replace Indic-script words using `mapping`, else unidecode. Latin text is de-accented."""
    if not INDIC.search(text):
        return unidecode(text)
    out = []
    for tok in re.split(r"(\s+|[,()/\-])", text):
        if INDIC.search(tok):
            key = tok.lower().strip(".")
            out.append(mapping.get(key) or unidecode(tok))
        else:
            out.append(unidecode(tok))
    return "".join(out)


DOTTED = re.compile(r"\b[a-z](?:\.\s?[a-z])+\.?(?![a-z])")  # p.c. / l.l.c. / c.i.t., not h.no


def _basic(text):
    text = text.lower().replace("&", " and ")
    text = DOTTED.sub(lambda m: re.sub(r"[.\s]", "", m.group(0)) + " ", text)  # l.l.c. -> llc
    text = re.sub(r"\bnull\b|\bnan\b|\bnone\b", " ", text)
    return SPACES.sub(" ", PUNCT.sub(" ", text)).strip()


def _name_tokens(raw):
    text = PAREN_COUNTRY.sub(lambda m: f" {PAREN_MARK}{m.group(1)} ", MS_PREFIX.sub(" ", raw))
    text = _basic(transliterate(text, _indic_dict()))
    text = LOOKALIKE.sub(lambda m: "l" if m.group(0) == "1" else "o", text)
    text = LOOKALIKE_START.sub(lambda m: "s" if m.group(1) == "5" else "b", text)
    # Roman numerals stay as-is in names: the data keeps them on both sides and has
    # look-alike noise ("XI" vs "Xl") that converting to digits would make worse.
    return [NAME_VARIANTS.get(t) or LEGAL.get(t, t) for t in text.split()]


def normalize_name(raw, country=""):
    """Return (name, name_core, name_key).

    name: full normalized name (alias markers removed, both parts kept)
    name_core: the entity's own name (text after an alias marker), legal forms/stopwords dropped
    name_key: name_core without generic business words (center, services, group, ...)
    """
    parts = ALIAS_SPLIT.split(raw)
    own = parts[-1] if len(parts) > 1 and parts[-1].strip(" :-,") else raw
    toks = _name_tokens(" ".join(parts))
    norm = " ".join(t[len(PAREN_MARK):] if t.startswith(PAREN_MARK) else t for t in toks)
    core = []
    for t in (_name_tokens(own) if own is not raw else toks):
        if t in LEGAL.values() or t in NAME_STOP or t.startswith(PAREN_MARK):
            continue
        if not core or core[-1] != t:  # "Mumbai Mumbai Finvest" -> "mumbai finvest"
            core.append(t)
    key = [t for t in core if t not in GENERIC]
    core_s = " ".join(core) or norm
    return norm, core_s, " ".join(key) or core_s


_KNOWN_ADDR = set(ADDR_COMMON) | set(ADDR_COMMON.values()) | set(STREET_TYPES)
_TYPO_TARGETS = [s for s in STREET_TYPES if len(s) >= 5 and s != "place"]  # not lane/road/quai


@lru_cache(maxsize=200_000)
def _street_typo(tok):
    """Misspelled street type -> canonical ("aveue"->"avenue", "stret"->"street"). One edit
    (incl. transposition), same first letter, words of 5+ letters only, so real street
    names like "Lance" or "Aveline" are left alone."""
    if len(tok) < 5 or not tok.isalpha() or tok in _KNOWN_ADDR:
        return tok
    for target in _TYPO_TARGETS:
        if tok[0] == target[0] and abs(len(tok) - len(target)) <= 1 and OSA.distance(tok, target) <= 1:
            return target
    return tok


def normalize_address(raw, country=""):
    text = raw.replace("°", " ").replace("º", " ")  # N° 45 -> N 45 (unidecode would give "deg")
    for native, eng in INDIC_STATES.items():
        text = text.replace(native, eng)
    states = STATE_COMPONENT.get(country, {})
    parts = [_basic(p) for p in transliterate(text, _indic_dict()).split(",")]
    text = " ".join(states.get(p, p) for p in parts if p)
    text = COMPOUND_NUM.sub(lambda m: str(_TENS[m.group(1)] + _UNITS[m.group(2)]), text)
    text = GLUED_SUFFIX.sub(r"\1 \2", text)
    local = ADDR_COUNTRY.get(country, {})
    india, france = country == "India", country == "France"
    toks = []
    for t in text.split():
        if france and t == "b" and toks and toks[-1].isdigit():
            t = "bis"                                     # 94 B -> 94 bis
        if india:
            t = IN_CITY.get(t) or IN_PREMISE.get(t, t)
        t = _street_typo(t)
        t = local.get(t) or ADDR_COMMON.get(t) or ROMAN.get(t) or ADDR_NUMWORDS.get(t, t)
        t = re.sub(r"^(\d+)(st|nd|rd|th)$", r"\1", t)  # 2nd -> 2, 45th -> 45
        t = re.sub(r"^0+(\d)", r"\1", t)                # 0049 -> 49, 0684 -> 684
        if t in ADDR_NOISE or (india and t == "no" and toks and toks[-1] == "no"):
            continue
        if france and t in ADDR_FR_DROP:
            continue
        toks.append(t)
    return " ".join(toks)


def _norm_rows(rows):
    out = []
    for eid, name, addr, country in rows:
        n, c, k = normalize_name(name, country)
        out.append((eid, n, c, k, normalize_address(addr, country), country))
    return out


def normalize_frame(df, pool, block=500_000):
    """Normalize in blocks so only one block's Python tuples live in memory at a time."""
    cols = ["entity_id", "business_name", "business_address", "country"]
    out = []
    for start in range(0, df.height, block):
        rows = df.slice(start, block).select(cols).rows()
        chunks = [rows[i:i + 20_000] for i in range(0, len(rows), 20_000)]
        res = [r for part in pool.map(_norm_rows, chunks) for r in part]
        out.append(pl.DataFrame(res, orient="row", schema=[
            "entity_id", "name", "name_core", "name_key", "addr", "country"]))
    return pl.concat(out)


def build_dict():
    """Learn Indic word -> English word by positional alignment of true-pair names."""
    pairs = read_pairs()
    s1 = read_source("train", 1, ["entity_id", "business_name"]).rename(
        {"entity_id": "s1", "business_name": "n1"})
    other = pl.concat([read_source("train", i, ["entity_id", "business_name"]) for i in (2, 3)])
    other = other.rename({"entity_id": "m", "business_name": "n2"}).filter(
        pl.col("n2").str.contains(r"[ऀ-෿]"))
    joined = pairs.join(other, on="m").join(s1, on="s1")
    counts = defaultdict(Counter)
    for n1, n2 in joined.select("n1", "n2").iter_rows():
        t1 = [t for t in SPLIT.split(n1.lower()) if t]
        t2 = [t for t in SPLIT.split(n2.lower()) if t]
        if len(t1) != len(t2):
            continue
        for eng, ind in zip(t1, t2):
            if INDIC.search(ind) and not INDIC.search(eng):
                counts[ind][unidecode(eng)] += 1
    mapping = {k: c.most_common(1)[0][0] for k, c in counts.items()}
    DICT_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"learned {len(mapping):,} Indic words from {joined.height:,} pairs -> {DICT_PATH}")


def main():
    arg = sys.argv[1]
    if arg == "build-dict":
        build_dict()
        return
    with Pool(14) as pool:
        for i in (1, 2, 3):
            df = normalize_frame(read_source(arg, i), pool)
            out = WORK / f"{arg}_s{i}_norm.parquet"
            df.write_parquet(out)
            print(f"{arg} source{i}: {df.height:,} rows -> {out.name}", flush=True)
            del df


if __name__ == "__main__":
    main()
