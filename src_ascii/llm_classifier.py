"""Claude Haiku validation AI Exposure Score.

Wywoluje Claude Haiku per spolka, podaje wycinki 10-K z AI mentions, parsuje
odpowiedz w formacie SCORE: 0-3 + REASONING. Zapisuje raw responses do jsonl.

Konfiguracja reprodukowalna: temperature=0, fixed model version, prompt z pliku.

Gracefully handles missing ANTHROPIC_API_KEY: zapisuje ai_exposure_llm=NaN
i kontynuuje (deterministic score jest specyfikacja glowna).

Usage:
    python src/llm_classifier.py            # try LLM, fallback if no key
    python src/llm_classifier.py --no-llm   # skip LLM completely
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from common import CACHE

SCORES_IO = CACHE / "ai_exposure_scores.csv"
RAW_OUT = CACHE / "llm_raw_responses.jsonl"
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "ai_exposure_llm_prompt.md"
META_TICKERS = CACHE / "metadata_tickers.csv"

MODEL_NAME = "claude-haiku-4-5"
TEMPERATURE = 0
MAX_TOKENS = 200
SCORE_TO_NORMALIZED = {0: 0.0, 1: 0.33, 2: 0.67, 3: 1.0}


def load_prompt() -> tuple[str, str]:
    """Wczytuje system + user prompt template z pliku Markdown."""
    text = PROMPT_PATH.read_text()
    sys_match = re.search(r"## System prompt\s*```\s*(.+?)```", text, re.DOTALL)
    user_match = re.search(r"## User prompt template\s*```\s*(.+?)```", text, re.DOTALL)
    if not sys_match or not user_match:
        raise ValueError(f"Nie znaleziono system/user promptu w {PROMPT_PATH}")
    return sys_match.group(1).strip(), user_match.group(1).strip()


def parse_response(text: str) -> tuple[int | None, str, bool]:
    m = re.search(r"SCORE:\s*([0-3])\s*\n+\s*REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if not m:
        m = re.search(r"SCORE:\s*([0-3])", text, re.IGNORECASE)
        if m:
            return int(m.group(1)), "", False
        return None, "", True
    return int(m.group(1)), m.group(2).strip(), False


def call_claude(client, system_prompt: str, user_prompt: str) -> str:
    resp = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text if resp.content else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="Pomin LLM validation calkowicie")
    args = parser.parse_args()

    scores_df = pd.read_csv(SCORES_IO)
    print(f"Wczytano {len(scores_df)} spolek z {SCORES_IO.name}")

    if args.no_llm:
        print("--no-llm: pomijam LLM. Zapisuje ai_exposure_llm=NaN.")
        scores_df["ai_exposure_llm"] = np.nan
        scores_df["llm_score_raw"] = np.nan
        scores_df.to_csv(SCORES_IO, index=False)
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("BRAK ANTHROPIC_API_KEY w env. Fallback: ai_exposure_llm=NaN, kontynuuje deterministic only.")
        scores_df["ai_exposure_llm"] = np.nan
        scores_df["llm_score_raw"] = np.nan
        scores_df.to_csv(SCORES_IO, index=False)
        return

    try:
        import anthropic
    except ImportError:
        print("Brak anthropic SDK. Fallback do NaN.")
        scores_df["ai_exposure_llm"] = np.nan
        scores_df["llm_score_raw"] = np.nan
        scores_df.to_csv(SCORES_IO, index=False)
        return

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt, user_template = load_prompt()
    prompt_sha = hashlib.sha256((system_prompt + user_template).encode()).hexdigest()[:16]
    print(f"Prompt SHA-256[:16]: {prompt_sha}; model: {MODEL_NAME}; T={TEMPERATURE}")

    tickers_meta = pd.read_csv(META_TICKERS).set_index("ticker")["company_name"].to_dict()

    raw_handle = open(RAW_OUT, "w", encoding="utf-8")
    llm_scores = []
    raw_scores = []

    for _, row in scores_df.iterrows():
        ticker = row["ticker"]
        excerpts_path = Path(row["excerpts_path"])
        if not excerpts_path.is_absolute():
            excerpts_path = CACHE.parent / excerpts_path
        if not excerpts_path.exists() or row["excerpts_char_count"] == 0:
            print(f"  [{ticker}] BRAK excerptow -> NaN")
            llm_scores.append(np.nan)
            raw_scores.append(np.nan)
            continue
        excerpts = excerpts_path.read_text(encoding="utf-8")[:12000]
        user_prompt = user_template.format(
            ticker=ticker,
            company_name=tickers_meta.get(ticker, ticker),
            filing_date=row["filing_date"],
            period_of_report=row["period_of_report"],
            excerpts=excerpts,
        )
        try:
            raw = call_claude(client, system_prompt, user_prompt)
            score, reasoning, parse_error = parse_response(raw)
            normalized = SCORE_TO_NORMALIZED.get(score, np.nan) if score is not None else np.nan
            print(f"  [{ticker}] SCORE={score} norm={normalized}")
            log = {
                "ticker": ticker,
                "model_name": MODEL_NAME,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "prompt_sha256_16": prompt_sha,
                "raw_response": raw,
                "parsed_score": score,
                "reasoning": reasoning,
                "ai_exposure_llm": normalized,
                "parse_error": parse_error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "excerpts_char_count": len(excerpts),
            }
            raw_handle.write(json.dumps(log, ensure_ascii=False) + "\n")
            llm_scores.append(normalized)
            raw_scores.append(score if score is not None else np.nan)
            time.sleep(0.1)
        except Exception as e:
            print(f"  [{ticker}] ERROR: {e}")
            llm_scores.append(np.nan)
            raw_scores.append(np.nan)

    raw_handle.close()
    scores_df["ai_exposure_llm"] = llm_scores
    scores_df["llm_score_raw"] = raw_scores
    scores_df.to_csv(SCORES_IO, index=False)

    valid = scores_df.dropna(subset=["ai_exposure_llm", "ai_exposure_det_main"])
    if len(valid) >= 5:
        rho = valid[["ai_exposure_det_main", "ai_exposure_llm"]].corr(method="spearman").iloc[0, 1]
        print(f"\nKorelacja Spearmana det_main vs llm: rho = {rho:.4f} (n={len(valid)})")
    else:
        print("\nZa malo waznych wynikow LLM zeby policzyc korelacje.")


if __name__ == "__main__":
    main()
