# Pre-rejestracja hipotez i decyzji metodologicznych

**Projekt**: AI Boom Premium — pre-event AI exposure, model risk and validated investment analytics
**Autorzy**: Roksana Kulawiec (454474), Sylwia Malinowska (487648), Mateusz Konopka (487528), Antoni Grabowski (452622)
**Data utworzenia**: 2026-05-20
**Status**: Spisane PRZED uruchomieniem centralnej cross-sectional regression (Etap 3 planu).

---

## 1. Pytanie badawcze

**RQ1**: Czy pre-event AI exposure, mierzona na podstawie raportów 10-K dostępnych przed 30.11.2022, wyjaśnia zróżnicowanie post-event abnormal returns wśród analizowanych spółek?

## 2. Hipotezy

- **H1 (główna)**: Spółki z wyższym pre-event AI Exposure Score osiągnęły wyższe abnormal returns po 30.11.2022.
- **H2**: Efekt AI exposure pozostaje dodatni po kontroli czynników ryzyka (beta, volatility, momentum, size).
- **H3**: Wynik nie jest w pełni napędzany przez pojedyncze obserwacje skrajne.
- **H4**: AI infrastructure portfolio osiąga dodatnią alfę również po kontroli klasycznych czynników asset pricing (market, size, value, momentum).

## 3. Próba i zakres danych

- **n = 41 spółek** z 4 grup (AI infrastructure, AI platforms, broad technology, defensive). Definicje grup: `data_cache/metadata_tickers.csv`.
- Cache `data_cache/prices.parquet` zawiera **43 instrumenty** (41 spółek + benchmark SPY + risk-free proxy `^IRX`).
- **Pre-event period**: 2021-01-01 → 2022-11-29.
- **Event date**: 2022-11-30 (publiczna premiera ChatGPT).
- **Post-event main window**: 2022-11-30 → 2023-05-31 (6 miesięcy).
- **Robustness windows**: 3 / 6 / 12 miesięcy.

## 4. Zmienna zależna (Y)

**Główna specyfikacja**: 6-miesięczny abnormal return spółki *i* względem SPY:
```
AR_i = sum(log(stock_i_close)/log(stock_i_close.shift(1))) - sum(log(SPY_close)/log(SPY_close.shift(1)))
```
dla okresu 2022-11-30 → 2023-05-31, suma dziennych log-zwrotów.

**Robustness**:
- CAPM-adjusted abnormal return (z pre-event beta).
- Carhart 4-factor residual return (alpha + ε z estymacji event-window).

## 5. Zmienna główna (X)

**Pre-event AI Exposure Score** zbudowana z najnowszego raportu 10-K SEC dostępnego publicznie **przed** 30.11.2022:

1. Filtr `filing_date < 2022-11-30`.
2. Ekstrakcja sekcji Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A).
3. Liczenie wzmianek wg słownika konserwatywnego (pre-2022 AI dictionary).
4. `mentions_per_10k_words = raw_count / (doc_length_words / 10000)`.
5. `log_score = log(1 + mentions_per_10k_words)`.
6. Winsorization 1%/99%.
7. **`ai_exposure_det_main = percentile_rank(log_score) ∈ [0, 1]`** — to jest specyfikacja główna.

**Słownik konserwatywny (main)**:
```
artificial intelligence, machine learning, deep learning, neural network,
natural language processing, computer vision, predictive analytics,
autonomous systems, recommendation system, data science, automation,
algorithmic decision-making
```

**Specjalne traktowanie `GPU`**: liczone tylko gdy w oknie ±50 słów występuje `AI / machine learning / training / inference / data center / accelerator / neural network / deep learning`.

## 6. Kontrole (controls)

Wszystkie liczone z okna **pre-event** (2021-01-01 → 2022-11-29):
- `Beta_Pre_i` — beta CAPM względem SPY.
- `Vol_Pre_i` — annualized volatility dziennych log-zwrotów.
- `Momentum_12M_Pre_i` — 12-miesięczny log-zwrot (od 2021-11-30 do 2022-11-29).
- `log_Market_Cap_i` — log of market cap na 2022-11-29 (źródło: yfinance fast_info lub fundamentalne).

## 7. Modele główne (pre-specified)

Estymacja OLS (statsmodels):
```
M1: AR_i = β0 + β1·ai_exposure_det_main_i + ε_i
M2: AR_i = β0 + β1·ai_exposure_det_main_i + β2·Beta_Pre_i + β3·Vol_Pre_i + ε_i
M3: + β4·Momentum_12M_Pre_i
M4: + β5·log_Market_Cap_i
```

**Świadomie nie dodajemy**:
- 11 GICS sector dummies (przeciążyłyby model z n=41),
- interakcji,
- nieliniowych przekształceń AI_Exposure (z wyjątkiem testu kwantyli w robustness).

## 8. Inference (pre-specified)

Dla każdego modelu raportujemy łącznie:
- współczynnik β₁ z 95% CI standardowym,
- White (HC1) robust SE,
- bootstrap 95% CI (B = 10 000, ziarno 42),
- ekonomiczną magnitudę: predicted post-event return przy AI_Exposure = 0 vs AI_Exposure = 1.

**Decision rule (nie używana automatycznie do narracji)**: efekt traktujemy za empirycznie wspierający H1 jeśli:
- β₁ > 0 we wszystkich M1-M4,
- 95% bootstrap CI dla β₁ w M3 lub M4 nie zawiera 0,
- wynik przeżywa leave-one-out (rozkład β₁ z LOO nie zmienia znaku w >5% przypadków).

## 9. Robustness checks (must-have, pre-specified)

1. **Leave-one-out (n = 41 regresji)** dla M3 — rozkład β₁ raportowany w `figures/leave_one_out_beta1.pdf`.
2. **Excluding extreme firms**: bez NVIDIA, bez top-1 AI exposure, bez top-3 AI exposure, bez top-3 post-event return.
3. **Alternative AI scores**: `det_main` (główny), `det_extended` (z rozszerzonym słownikiem), `llm` (Claude Haiku), kompozyt (eksploracyjnie).
4. **Alternative windows Y**: 3 / 6 (główny) / 12 miesięcy.
5. **Alternative scaling X**: percentile rank (główny) / z-score / raw mentions per 10k / log score.

## 10. Walidacja przez modele czynnikowe (Hipoteza H4)

Per spółka i per portfel grupowy, na dziennych danych pre-event do końca dostępnego okresu:
```
CAPM:      r_i - rf = α + β_mkt·(Mkt-RF) + ε
FF3:       + β_smb·SMB + β_hml·HML
Carhart 4: + β_mom·MOM
```
**Newey-West HAC SE** (lag = 5).
Dane czynników: **Kenneth French Data Library** (daily). RF z serii FF, nie z `^IRX`.

**Decision rule H4**: AI infrastructure portfolio uznajemy za posiadające trwałe abnormal returns jeśli α z Carhart 4 jest istotnie dodatnia przy `p < 0.10` (Newey-West).

## 11. LLM jako walidacja (nie main)

- Claude Haiku (model name + version zapisane), `temperature = 0`.
- Prompt z pliku `prompts/ai_exposure_llm_prompt.md` (SHA-256 zapisany w outputach).
- Raw responses → `data_cache/llm_raw_responses.jsonl`.
- Parsed score → `data_cache/ai_exposure_scores.csv` kolumna `ai_exposure_llm`.
- Raportujemy korelację Spearmana `det_main` vs `llm` — wysoka korelacja wspiera wiarygodność miary.
- LLM **nie wchodzi** do głównego scoru jako arbitralny mix (np. 0.6/0.4).

## 12. Co świadomie pomijamy

- **Difference-in-Differences w pełnym panelu** — przy endogenicznym doborze grupy AI (definicja) brak parallel trends.
- **Cross-asset transmission** (power, copper, uranium) — poza zakresem.
- **Synthetic control** — opcjonalny, nie MVP.
- **Predykcja przyszłej AI premium** — analiza jest *ex post* nie *ex ante*.

## 13. Co może wymagać post-hoc decyzji (i zostanie odnotowane w decision_log.md)

- Wybór konkretnego 10-K dla spółek o niestandardowym fiscal year-end.
- Wykluczenie spółek bez 10-K przed 2022-11-30.
- Wybór konkretnej wersji Claude Haiku (jeśli zostanie zaktualizowany w trakcie projektu).
- Obsługa missing market cap (yfinance czasem nie zwraca).
- Wybór konkretnego pliku Fama-French (daily vs monthly; standardowy vs research).

## 14. Pre-commitment do raportowania wyników

Zobowiązujemy się raportować w finalnym raporcie:
- **Wszystkie** modele M1–M4 (nie tylko ten najsilniejszy).
- **Wszystkie** alternatywne specyfikacje z sekcji 9 (nie tylko te potwierdzające H1).
- Wszelkie post-hoc zmiany metodologii względem tej pre-rejestracji (w decision_log.md + sekcji „Deviations from pre-registration" w raporcie).
- Wynik LOO nawet jeśli pokaże niestabilność.
- Wartości p ≥ 0.05 traktujemy jako pełnoprawne wyniki (nie jako „brak wyniku").
