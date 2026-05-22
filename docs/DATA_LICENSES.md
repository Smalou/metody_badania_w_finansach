# Data sources and licenses

## Yahoo Finance (przez yfinance)
- **Co używamy**: dzienne ceny zamknięcia (Adj Close) dla 41 spółek + SPY + ^IRX (2021-01-01 → 2026-05-14).
- **Cache**: `data_cache/prices.parquet`.
- **Licencja**: Yahoo Finance Terms of Use — dane są darmowe do użytku niekomercyjnego, w tym akademickiego. Yahoo nie gwarantuje accuracy/timeliness. Dane mogą być korygowane wstecznie.
- **Cytowanie**: Yahoo Finance via `yfinance` (Aroussi, R.), https://github.com/ranaroussi/yfinance.

## SEC EDGAR
- **Co używamy**: pełne teksty raportów 10-K (form type 10-K) dla 41 spółek, najnowsze złożone przed 2022-11-30.
- **Cache**: `data_cache/10k_filings/` (HTML/TXT) + `data_cache/10k_metadata.csv`.
- **Licencja**: Public domain (rządowe dokumenty USA, 17 U.S.C. § 105). Wolny od ograniczeń copyright.
- **Wymagania techniczne**: User-Agent z emailem grupy projektowej (SEC fair access policy), rate limit 10 req/s.
- **Cytowanie**: U.S. Securities and Exchange Commission, EDGAR system, https://www.sec.gov/edgar.

## Kenneth French Data Library
- **Co używamy**: dzienne czynniki Fama-French 3-factor + Momentum (Mkt-RF, SMB, HML, MOM, RF).
- **Cache**: `data_cache/ff_factors.csv`.
- **Licencja**: Free for academic and non-commercial use; nie dla rebrandingu/redystrybucji komercyjnej. Standard cytowanie w pracach finansowych.
- **Cytowanie**: Kenneth R. French Data Library, Tuck School of Business at Dartmouth, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html.
- **Podstawowe odniesienia teoretyczne**: Fama & French (1993), Carhart (1997) — pełne w `report/refs.bib`.

## Anthropic Claude API (Haiku)
- **Co używamy**: klasyfikacja AI exposure na podstawie wycinków 10-K.
- **Model**: `claude-haiku-4-5` (lub aktualny stable Haiku w czasie uruchomienia).
- **Konfiguracja**: temperature=0, fixed system prompt.
- **Cache**: `data_cache/llm_raw_responses.jsonl` (każda linia: ticker, model_version, prompt_sha256, raw_response, parsed_score, timestamp).
- **Licencja**: Anthropic Usage Policies. API key jest płatny (per-token), własność użytkownika. Output (raw text) może być używany dla analizy własnej.
- **Reprodukowalność**: Anthropic okresowo aktualizuje modele. Cached `llm_raw_responses.jsonl` w repo umożliwia reprodukcję bez ponownego wywołania API. Skrypt obsługuje też `--no-llm` flag dla pełnej reprodukcji bez API key.
- **Cytowanie**: Anthropic (2024-2026). Claude Haiku, https://www.anthropic.com.

## Dane wzbogacające (opcjonalne)
- **Market cap na 2022-11-29**: yfinance `Ticker.fast_info.market_cap` (snapshot daty pobrania, nie historyczny). Może odbiegać od rzeczywistej wartości na 2022-11-29 dla niektórych spółek; alternatywą jest SEC EDGAR XBRL fundamentals (poza scope MVP).

---

## Compliance i etyka

- Wszystkie dane są publicznie dostępne, brak danych osobowych.
- Brak insider trading data, brak proprietary research.
- Repozytorium kodu na licencji MIT (`LICENSE-CODE`); dokumentacja i raport na CC-BY-4.0 (`LICENSE-DOCS`).
