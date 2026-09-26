"""Text normalisation: Indic -> Latin transliteration, accent stripping, punctuation cleanup."""
import re

import polars as pl

# The Brahmic Unicode blocks (Devanagari 0x0900 .. Malayalam 0x0D00) share one layout,
# so a single offset table transliterates all of them.
_CONS = dict(zip(range(0x15, 0x3A), (
    "k kh g gh n ch chh j jh n t th d dh n t th d dh n n p ph b bh m y r r l l zh v sh sh s h").split()))
_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
           0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
           0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au"}
_ABS = {0x09CE: "t", 0x09F0: "r", 0x09F1: "w", 0x0A70: "n", 0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r",
        0x0D7D: "l", 0x0D7E: "l"}
_VIRAMA, _NUKTA = 0x4D, 0x3C
INDIC_RE = r"[ऀ-ൿ]"
_INDIC = re.compile(INDIC_RE)


def transliterate(s: str) -> str:
    """Phonetic Indic->Latin. Inherent 'a' is dropped at word end (schwa deletion)."""
    if not s or not _INDIC.search(s):
        return s
    out, pending = [], False
    for ch in s:
        o = ord(ch)
        if o in _ABS:
            out.append(_ABS[o]); pending = False
            continue
        if not 0x0900 <= o < 0x0D80:
            pending = False
            out.append(ch)
            continue
        k = (o - 0x0900) & 0x7F
        if k in _CONS:
            if pending:
                out.append("a")
            out.append(_CONS[k]); pending = True
        elif k in _MATRAS:
            out.append(_MATRAS[k]); pending = False
        elif k == _VIRAMA:
            pending = False
        elif k == _NUKTA:
            pass
        else:
            if pending:
                out.append("a"); pending = False
            if k in _VOWELS:
                out.append(_VOWELS[k])
            elif k in (0x01, 0x02):
                out.append("n")
            elif k == 0x03:
                out.append("h")
            elif 0x66 <= k <= 0x6F:
                out.append(str(k - 0x66))
            else:
                out.append(" ")
    return "".join(out)


def normalize_expr(col: str) -> pl.Expr:
    """Vectorised cleanup; run after `transliterate`. Country-agnostic by design."""
    e = pl.col(col).fill_null("")
    e = e.str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
    e = e.str.replace(r"^[^a-z0-9]*(?:www\.)?([a-z0-9-]+)\.[a-z]{2,6}(?:\.[a-z]{2})?[^a-z0-9]*$", "$1")  # domain-form names
    e = e.str.replace_all(r"\bn/a\b", " ").str.replace_all("&", " and ")
    e = e.str.replace_all(r"[^\p{L}\p{N}]+", " ")
    e = e.str.replace_all(r"\b0+(\d)", "$1")  # 004013 -> 4013
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


def normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Adds name_n, addr_n, name_c (compact, no spaces) and has_indic."""
    df = df.with_columns(
        has_indic=pl.col("name").str.contains(INDIC_RE) | pl.col("addr").str.contains(INDIC_RE))
    df = df.with_columns([
        pl.when(pl.col("has_indic")).then(pl.col(c).map_elements(transliterate, return_dtype=pl.String))
        .otherwise(pl.col(c)).alias(c) for c in ("name", "addr")
    ]) if df["has_indic"].any() else df
    return df.with_columns(name_n=normalize_expr("name"), addr_n=normalize_expr("addr")).with_columns(
        name_c=pl.col("name_n").str.replace_all(" ", ""))


if __name__ == "__main__":
    assert transliterate("राम मार्केटिंग प्राइवेट लिमिटेड") == "ram marketing praivet limited", transliterate("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert transliterate("abc") == "abc"
    t = pl.DataFrame({"name": ["SIMONARTIFICIAL.COM", "M/s Dark Clinic & Co.", "Cornerstone Cóastal", "*** compassutility.com"],
                      "addr": ["004013 Valance Way", "MNAV - 29, N/A, West Bengal", None, ""]})
    r = normalize_frame(t)
    assert r["name_n"].to_list() == ["simonartificial", "m s dark clinic and co", "cornerstone coastal", "compassutility"], r["name_n"].to_list()
    assert r["addr_n"].to_list() == ["4013 valance way", "mnav 29 west bengal", "", ""], r["addr_n"].to_list()
    print("normalize self-test OK")
