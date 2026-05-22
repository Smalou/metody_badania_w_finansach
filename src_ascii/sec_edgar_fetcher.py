"""Pobranie najnowszych 10-K dostepnych przed 2022-11-30 dla 41 spolek z metadata_tickers.csv.

SEC EDGAR API: https://data.sec.gov/ + https://www.sec.gov/cgi-bin/browse-edgar
Wymaga User-Agent z emailem (SEC fair access policy).

Cache:
- data_cache/10k_filings/{ticker}_{accession}.txt
- data_cache/10k_metadata.csv
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

from common import CACHE, load_metadata

CUTOFF_DATE = datetime(2022, 11, 30).date()
USER_AGENT = "Sylwia Malinowska (malinowska0sylwia@gmail.com) - Metody Badan w Finansach research project"

FILINGS_DIR = CACHE / "10k_filings"
FILINGS_DIR.mkdir(exist_ok=True)
META_OUT = CACHE / "10k_metadata.csv"
TICKERS_MAP = CACHE / "sec_tickers.json"

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FILING_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})

RATE_LIMIT_SLEEP = 0.12  # SEC pozwala ~10 req/s


def _sleep():
    time.sleep(RATE_LIMIT_SLEEP)


def fetch_ticker_to_cik_map() -> dict[str, str]:
    """Pobiera company_tickers.json z SEC i zwraca map ticker -> 10-digit CIK."""
    if TICKERS_MAP.exists():
        return json.loads(TICKERS_MAP.read_text())
    print(f"Pobieram mapping ticker->CIK z {SEC_TICKERS_URL}")
    r = SESSION.get(SEC_TICKERS_URL)
    r.raise_for_status()
    data = r.json()
    # format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    mapping = {row["ticker"]: f"{row['cik_str']:010d}" for row in data.values()}
    TICKERS_MAP.write_text(json.dumps(mapping, indent=2))
    _sleep()
    return mapping


def fetch_company_submissions(cik: str) -> dict:
    """Pobiera submissions metadata dla danego CIK (10-digit, z leading zeros)."""
    url = SUBMISSIONS_URL.format(cik=cik)
    r = SESSION.get(url)
    r.raise_for_status()
    _sleep()
    return r.json()


def _scan_recent_block(block: dict, target_forms: set[str], cutoff) -> dict | None:
    """Iteruje przez blok recent / paginated i wybiera najnowszy form ze zbioru target_forms przed cutoff."""
    forms = block.get("form", [])
    filing_dates = block.get("filingDate", [])
    accession_numbers = block.get("accessionNumber", [])
    primary_docs = block.get("primaryDocument", [])
    primary_doc_descs = block.get("primaryDocDescription", [])
    period_of_report = block.get("reportDate", [])
    best = None
    for i, form in enumerate(forms):
        if form not in target_forms:
            continue
        try:
            fdate = datetime.strptime(filing_dates[i], "%Y-%m-%d").date()
        except (ValueError, IndexError):
            continue
        if fdate >= cutoff:
            continue
        candidate = {
            "form_type": form,
            "filing_date": fdate.isoformat(),
            "accession_number": accession_numbers[i],
            "primary_document": primary_docs[i],
            "primary_doc_desc": primary_doc_descs[i] if i < len(primary_doc_descs) else "",
            "period_of_report": period_of_report[i] if i < len(period_of_report) else "",
        }
        if best is None or candidate["filing_date"] > best["filing_date"]:
            best = candidate
    return best


def _fetch_older_files(submissions: dict) -> list[dict]:
    """Pobiera starsze (paginated) submissions z submissions.filings.files."""
    files_meta = submissions.get("filings", {}).get("files", [])
    blocks = []
    for fm in files_meta:
        name = fm.get("name")
        if not name:
            continue
        url = f"https://data.sec.gov/submissions/{name}"
        r = SESSION.get(url)
        if r.status_code != 200:
            continue
        _sleep()
        blocks.append(r.json())
    return blocks


def find_latest_10k_before_cutoff(submissions: dict, cutoff=CUTOFF_DATE) -> dict | None:
    """Najnowsze 10-K przed cutoff, ze sprawdzeniem rowniez paginated older files."""
    best = _scan_recent_block(submissions.get("filings", {}).get("recent", {}), {"10-K"}, cutoff)
    if best is not None:
        return best
    # fallback: starsze paginated blocks
    for block in _fetch_older_files(submissions):
        candidate = _scan_recent_block(block, {"10-K"}, cutoff)
        if candidate is not None:
            if best is None or candidate["filing_date"] > best["filing_date"]:
                best = candidate
    return best


def find_latest_20f_before_cutoff(submissions: dict, cutoff=CUTOFF_DATE) -> dict | None:
    """Foreign filers (TSM, ASML) skladaja 20-F zamiast 10-K. Fallback."""
    best = _scan_recent_block(submissions.get("filings", {}).get("recent", {}), {"20-F", "20-F/A"}, cutoff)
    if best is not None:
        return best
    for block in _fetch_older_files(submissions):
        candidate = _scan_recent_block(block, {"20-F", "20-F/A"}, cutoff)
        if candidate is not None:
            if best is None or candidate["filing_date"] > best["filing_date"]:
                best = candidate
    return best


def download_filing_text(cik: str, accession: str, primary_doc: str, ticker: str,
                         force: bool = False) -> Path:
    """Pobiera primary document i konwertuje do plain text. Auto-detect XML vs HTML.

    Wiele nowoczesnych 10-K to inline XBRL (iXBRL) zapisany jako XML; lxml HTML
    parser nie wyciaga z nich tekstu. Detekcja po pierwszych bajtach.
    """
    accession_nodash = accession.replace("-", "")
    cik_int = int(cik)
    out_path = FILINGS_DIR / f"{ticker}_{accession_nodash}.txt"
    if out_path.exists() and not force:
        return out_path
    url = FILING_BASE.format(cik_int=cik_int, accession_nodash=accession_nodash, primary_doc=primary_doc)
    r = SESSION.get(url)
    r.raise_for_status()
    head = r.content[:200].lstrip().lower()
    parser = "xml" if head.startswith(b"<?xml") else "lxml"
    soup = BeautifulSoup(r.content, parser)
    for s in soup(["script", "style"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join([ln for ln in lines if ln])
    out_path.write_text(text, encoding="utf-8")
    _sleep()
    return out_path


def main():
    meta = load_metadata()
    tickers = meta["ticker"].tolist()
    print(f"Pobieram 10-K (oraz 20-F dla foreign filers) dla {len(tickers)} spolek; cutoff={CUTOFF_DATE}")

    cik_map = fetch_ticker_to_cik_map()

    rows = []
    for ticker in tickers:
        if ticker not in cik_map:
            print(f"  [{ticker}] BRAK w sec_tickers.json -- pomijam")
            rows.append({
                "ticker": ticker, "cik": None, "form_type": None, "filing_date": None,
                "period_of_report": None, "accession_number": None, "primary_document": None,
                "source_url": None, "filepath": None, "download_timestamp": None,
                "status": "no_cik_mapping",
            })
            continue
        cik = cik_map[ticker]
        try:
            subs = fetch_company_submissions(cik)
            best = find_latest_10k_before_cutoff(subs)
            if best is None:
                best = find_latest_20f_before_cutoff(subs)
                if best is None:
                    print(f"  [{ticker}] BRAK 10-K i 20-F przed {CUTOFF_DATE}")
                    rows.append({
                        "ticker": ticker, "cik": cik, "form_type": None, "filing_date": None,
                        "period_of_report": None, "accession_number": None, "primary_document": None,
                        "source_url": None, "filepath": None, "download_timestamp": None,
                        "status": "no_10k_or_20f_before_cutoff",
                    })
                    continue
                else:
                    print(f"  [{ticker}] foreign filer - uzywam 20-F ({best['filing_date']})")
            filepath = download_filing_text(cik, best["accession_number"], best["primary_document"], ticker)
            url = FILING_BASE.format(
                cik_int=int(cik),
                accession_nodash=best["accession_number"].replace("-", ""),
                primary_doc=best["primary_document"],
            )
            rows.append({
                "ticker": ticker, "cik": cik, "form_type": best["form_type"],
                "filing_date": best["filing_date"], "period_of_report": best["period_of_report"],
                "accession_number": best["accession_number"], "primary_document": best["primary_document"],
                "source_url": url, "filepath": str(filepath.relative_to(CACHE.parent)),
                "download_timestamp": datetime.utcnow().isoformat() + "Z", "status": "ok",
            })
            print(f"  [{ticker}] OK - {best['form_type']} z {best['filing_date']} -> {filepath.name}")
        except requests.HTTPError as e:
            print(f"  [{ticker}] HTTP error: {e}")
            rows.append({
                "ticker": ticker, "cik": cik, "form_type": None, "filing_date": None,
                "period_of_report": None, "accession_number": None, "primary_document": None,
                "source_url": None, "filepath": None, "download_timestamp": None,
                "status": f"http_error:{e}",
            })

    df = pd.DataFrame(rows)
    df.to_csv(META_OUT, index=False)
    n_ok = (df["status"] == "ok").sum()
    print(f"\nMetadata zapisana: {META_OUT}")
    print(f"OK: {n_ok}/{len(df)} spolek")
    if n_ok < len(df):
        missing = df[df["status"] != "ok"][["ticker", "status"]]
        print(f"Brakujace/problematyczne:\n{missing.to_string(index=False)}")


if __name__ == "__main__":
    main()
