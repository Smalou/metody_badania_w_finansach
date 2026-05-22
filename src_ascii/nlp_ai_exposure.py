"""Deterministic AI Exposure Score z 10-K tekstow.

Slownik glowny (conservative pre-2022) jako specyfikacja glowna.
Slownik rozszerzony jako robustness.
GPU context-conditional (tylko gdy w +/-50 slow jest term AI-related).
Normalizacja: mentions_per_10k_words -> log -> winsorize -> percentile_rank.

Output:
- data_cache/ai_exposure_scores.csv
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

from common import CACHE

FILINGS_DIR = CACHE / "10k_filings"
META_IN = CACHE / "10k_metadata.csv"
META_TICKERS = CACHE / "metadata_tickers.csv"
SCORES_OUT = CACHE / "ai_exposure_scores.csv"
EXCERPTS_DIR = CACHE / "10k_excerpts"
EXCERPTS_DIR.mkdir(exist_ok=True)

# ====================================================================
# SLOWNIKI
# ====================================================================
DICTIONARY_MAIN = [
    "artificial intelligence", "machine learning", "deep learning",
    "neural network", "natural language processing", "computer vision",
    "predictive analytics", "autonomous systems", "recommendation system",
    "data science", "automation", "algorithmic decision-making",
]

# Wariant robustness: bez "automation" (szerokie pojecie, false positives w 10-K
# moga oznaczac automatyzacje procesowa/operacyjna bez zwiazku z AI).
DICTIONARY_MAIN_NO_AUTOMATION = [t for t in DICTIONARY_MAIN if t != "automation"]

DICTIONARY_EXTENDED = [
    "generative ai", "large language model", "foundation model",
    "transformer model", "inference", "training compute", "ai accelerator",
]

# GPU jest liczone tylko jezeli w oknie +/-50 slow jest ktorys z tych terminow
GPU_CONTEXT_TERMS = [
    "ai", "artificial intelligence", "machine learning", "training",
    "inference", "data center", "accelerator", "neural network",
    "deep learning",
]

# ====================================================================
# CZYSZCZENIE 10-K
# ====================================================================
# 10-K w formacie inline XBRL zawiera duzy blok metadanych XBRL na poczatku
# i koncu pliku. Dodatkowo regex na "Item 1" jest niepewny ze wzgledu na
# spis tresci. Strategia: znajdz najpozniejsze wystapienie "Item 1." z dlugim
# tekstem po nim (= faktyczna sekcja); fallback = caly tekst po wycieciu
# najwczesniejszego XBRL bloku.
PART_I_PATTERN = re.compile(r"\bPART\s+I\b", re.IGNORECASE)


def _is_xbrl_noise_line(line: str) -> bool:
    """Heurystyka: linia z samych XBRL tagow / cyferek / dat."""
    if len(line) < 3 or len(line) > 200:
        return False
    # tylko cyfry/myslnki/kropki/dwukropki/separatory
    if re.match(r"^[\d\-:./T\sZ]+$", line):
        return True
    # XBRL/SGML tagi typu us-gaap: lub iso4217:
    if re.match(r"^[a-z][a-z0-9\-]*:[A-Za-z][A-Za-z0-9]*(Member)?$", line):
        return True
    # CIK / accession patterns
    if re.match(r"^\d{10}$", line) or re.match(r"^\d{4}-\d{2}-\d{2}$", line):
        return True
    return False


def clean_xbrl_noise(text: str) -> str:
    """Usuwa linie z XBRL/SGML metadanych, ktore zaclamuja statystyki dlugosci dokumentu."""
    lines = text.splitlines()
    cleaned = [ln for ln in lines if not _is_xbrl_noise_line(ln)]
    return "\n".join(cleaned)


def extract_relevant_sections(text: str) -> str:
    """Strategia uproszczona: caly tekst po wycieciu XBRL/SGML noise.

    Heurystyki na "PART I" / "Item 1." sa zawodne z powodu niejednorodnosci
    formatu 10-K (HTML / iXBRL / XML / mixed). W konsekwencji uzywamy calego
    oczyszczonego tekstu jako podstawy scoringu - to nie wprowadza biasu, bo
    interesuja nas relatywne dlugosci dokumentow (mentions per 10k words).
    """
    return clean_xbrl_noise(text)


# ====================================================================
# COUNTING
# ====================================================================
def tokenize_words(text: str) -> list[str]:
    """Lower + alphanumeric tokens."""
    return re.findall(r"[A-Za-z][A-Za-z0-9\-]*", text.lower())


def count_phrase_occurrences(text: str, phrase: str) -> int:
    """Liczy wystapienia frazy (case-insensitive, word-boundary)."""
    p = re.escape(phrase.lower())
    pattern = r"\b" + p + r"\b"
    return len(re.findall(pattern, text.lower()))


def count_gpu_with_context(text: str, window: int = 50) -> int:
    """Liczy GPU tylko jezeli w oknie +/-`window` slow wystepuje term AI-related."""
    words = tokenize_words(text)
    gpu_positions = [i for i, w in enumerate(words) if w == "gpu" or w == "gpus"]
    if not gpu_positions:
        return 0
    ctx_terms_lower = set(t.lower() for t in GPU_CONTEXT_TERMS if " " not in t)
    multi_word_ctx = [t.lower() for t in GPU_CONTEXT_TERMS if " " in t]

    count = 0
    for pos in gpu_positions:
        lo = max(0, pos - window)
        hi = min(len(words), pos + window + 1)
        window_words = words[lo:hi]
        # single-word check
        if any(w in ctx_terms_lower for w in window_words):
            count += 1
            continue
        # multi-word check (jako podlancuch zlaczonych slow)
        window_text = " ".join(window_words)
        if any(t in window_text for t in multi_word_ctx):
            count += 1
    return count


def count_total_mentions(text: str, dictionary: list[str], include_gpu: bool = False) -> tuple[int, dict[str, int]]:
    counts = {}
    for phrase in dictionary:
        counts[phrase] = count_phrase_occurrences(text, phrase)
    total = sum(counts.values())
    if include_gpu:
        gpu_count = count_gpu_with_context(text)
        counts["gpu (context)"] = gpu_count
        total += gpu_count
    return total, counts


# ====================================================================
# SCORING
# ====================================================================
def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def compute_score(mentions: pd.Series, doc_len: pd.Series) -> pd.DataFrame:
    """Konwertuje raw mentions + dlugosc dokumentu na percentile rank score."""
    mentions_per_10k = mentions / (doc_len / 10000)
    log_score = np.log1p(mentions_per_10k)
    log_score_w = winsorize(log_score)
    rank = log_score_w.rank(method="average") / len(log_score_w)
    return pd.DataFrame({
        "raw_mentions": mentions,
        "doc_length_words": doc_len,
        "mentions_per_10k": mentions_per_10k,
        "log_score": log_score,
        "log_score_winsorized": log_score_w,
        "percentile_rank": rank,
    })


def select_excerpts_for_llm(text: str, dictionary: list[str], max_chars: int = 12000,
                             context_window: int = 80) -> str:
    """Wybiera zdania zawierajace AI-related termy + kontekst dla GPU dla LLM input."""
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    selected = []
    used = set()
    for term in dictionary + ["gpu", "gpus"]:
        pat = re.compile(r"\b" + re.escape(term.lower()) + r"\b", re.IGNORECASE)
        for i, s in enumerate(sentences):
            if pat.search(s) and i not in used:
                # bierzemy +/-1 sasiadow dla kontekstu
                lo = max(0, i - 1)
                hi = min(len(sentences), i + 2)
                excerpt = " ".join(sentences[lo:hi])
                selected.append(excerpt)
                for j in range(lo, hi):
                    used.add(j)
                if sum(len(e) for e in selected) > max_chars:
                    break
        if sum(len(e) for e in selected) > max_chars:
            break
    return "\n\n---\n\n".join(selected)[:max_chars]


# ====================================================================
# MAIN
# ====================================================================
def main():
    meta = pd.read_csv(META_IN)
    meta = meta[meta["status"] == "ok"].copy()
    tickers_meta = pd.read_csv(META_TICKERS)
    meta = meta.merge(tickers_meta[["ticker", "group"]], on="ticker", how="left")
    print(f"Wczytano metadata dla {len(meta)} spolek z plikow 10-K")

    rows = []
    for _, row in meta.iterrows():
        ticker = row["ticker"]
        filepath = Path(row["filepath"])
        if not filepath.is_absolute():
            filepath = CACHE.parent / filepath
        if not filepath.exists():
            print(f"  [{ticker}] BRAK pliku: {filepath}")
            continue
        text = filepath.read_text(encoding="utf-8", errors="replace")
        sections = extract_relevant_sections(text)
        words = tokenize_words(sections)
        doc_len = len(words)

        total_main, counts_main = count_total_mentions(sections, DICTIONARY_MAIN, include_gpu=False)
        total_main_with_gpu, _ = count_total_mentions(sections, DICTIONARY_MAIN, include_gpu=True)
        total_main_no_aut, _ = count_total_mentions(sections, DICTIONARY_MAIN_NO_AUTOMATION, include_gpu=False)
        total_ext, _ = count_total_mentions(sections, DICTIONARY_EXTENDED, include_gpu=False)

        excerpts = select_excerpts_for_llm(sections, DICTIONARY_MAIN + DICTIONARY_EXTENDED)
        excerpts_path = EXCERPTS_DIR / f"{ticker}_excerpts.txt"
        excerpts_path.write_text(excerpts, encoding="utf-8")

        rows.append({
            "ticker": ticker,
            "group": row["group"],
            "form_type": row["form_type"],
            "filing_date": row["filing_date"],
            "period_of_report": row["period_of_report"],
            "doc_length_words": doc_len,
            "mentions_main": total_main,
            "mentions_main_with_gpu": total_main_with_gpu,
            "mentions_main_no_automation": total_main_no_aut,
            "mentions_extended": total_ext,
            "excerpts_path": str(excerpts_path.relative_to(CACHE.parent)),
            "excerpts_char_count": len(excerpts),
        })
        print(f"  [{ticker}] doc={doc_len:>6d}w; main={total_main:>3d}; +GPU={total_main_with_gpu:>3d}; no_auto={total_main_no_aut:>3d}; ext={total_ext:>3d}")

    df = pd.DataFrame(rows)
    main_scores = compute_score(df["mentions_main"], df["doc_length_words"])
    main_gpu_scores = compute_score(df["mentions_main_with_gpu"], df["doc_length_words"])
    main_no_aut_scores = compute_score(df["mentions_main_no_automation"], df["doc_length_words"])
    ext_scores = compute_score(df["mentions_extended"], df["doc_length_words"])

    df["mentions_per_10k_main"] = main_scores["mentions_per_10k"]
    df["mentions_per_10k_main_gpu"] = main_gpu_scores["mentions_per_10k"]
    df["mentions_per_10k_main_no_automation"] = main_no_aut_scores["mentions_per_10k"]
    df["mentions_per_10k_extended"] = ext_scores["mentions_per_10k"]
    df["ai_exposure_det_main"] = main_scores["percentile_rank"]
    df["ai_exposure_det_main_gpu"] = main_gpu_scores["percentile_rank"]
    df["ai_exposure_det_main_no_automation"] = main_no_aut_scores["percentile_rank"]
    df["ai_exposure_det_extended"] = ext_scores["percentile_rank"]

    df = df[[
        "ticker", "group", "form_type", "filing_date", "period_of_report",
        "doc_length_words",
        "mentions_main", "mentions_main_with_gpu", "mentions_main_no_automation", "mentions_extended",
        "mentions_per_10k_main", "mentions_per_10k_main_gpu",
        "mentions_per_10k_main_no_automation", "mentions_per_10k_extended",
        "ai_exposure_det_main", "ai_exposure_det_main_gpu",
        "ai_exposure_det_main_no_automation", "ai_exposure_det_extended",
        "excerpts_path", "excerpts_char_count",
    ]]
    df.to_csv(SCORES_OUT, index=False)
    print(f"\nZapisano: {SCORES_OUT} ({len(df)} spolek)")

    print("\nSrednie ai_exposure_det_main per grupa:")
    print(df.groupby("group")["ai_exposure_det_main"].agg(["count", "mean", "std", "min", "max"]).round(3))

    print("\nTop 10 spolek wedlug ai_exposure_det_main:")
    print(df.sort_values("ai_exposure_det_main", ascending=False)[
        ["ticker", "group", "mentions_main", "mentions_per_10k_main", "ai_exposure_det_main"]
    ].head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
