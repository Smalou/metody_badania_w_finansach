# Projekt zaliczeniowy IAD — ćwiczenia 3

Analiza statystyczna na danych rynkowych z **yfinance**: 41 spółek w czterech grupach tematycznych (AI infrastructure, AI platforms, broad tech, defensive), benchmark **SPY** oraz stopa wolna od ryzyka **^IRX** (w `src/data_fetch.py` domyślnie okres **2021-01-01 → 2026-05-15**).  
Przypisanie tickerów do grup i krótkie uzasadnienia: `data_cache/metadata_tickers.csv`.

**Deliverable:** `report/raport.pdf` (źródło: `report/raport.tex`).

---

## Zadania w kodzie

1. **Zadanie 1** (`src/task1_two_portfolios.py`) — po dacie zdarzenia **2022-11-30** (start boomu generatywnej AI / premiera ChatGPT) porównanie **średnich miesięcznych abnormal returns** (vs SPY) dwóch portfeli **równych wag (EW)**: **AI_infra** vs **Defensive** (kontrola).  
   Główny test: **Welch t-test** (nierówne wariancje); **bootstrap** 95% CI różnicy średnich; pomocniczo **Mann-Whitney U**; efekt: **Hedges’ g**.

2. **Zadanie 2** (`src/task2_four_groups.py`) — dla każdej spółki metryki m.in. zwrot i zmienność annualizowane, **Sharpe**, Sortino, max drawdown, **beta i alfa CAPM** względem SPY (z RF dziennym).  
   Główna zmienna: **Sharpe**; dodatkowa: **alfa CAPM**.  
   Wybór procedury: normalność (Shapiro) + homogeniczność wariancji (Levene) → **ANOVA + Tukey**, **Welch ANOVA + Games-Howell**, albo **Kruskal-Wallis + Dunn (Bonferroni)**.

3. **Zadanie 3** (`src/task3_paired_event.py`) — **A/B na parach zależnych**: 21 spółek (**AI_infra + AI_platforms**), dla każdej średnia dzienna AR w oknie **przed** vs **po** zdarzeniu **2022-11-30**; okna **30 / 60 / 120 / 250** sesji; główny wynik raportowany dla **120** sesji.  
   Test zależny od normalności różnic (Shapiro): **paired t-test** lub **Wilcoxon**; placebo na **2021-11-30**.

Opcjonalnie: **wykresy uzupełniające** — `src/plots_extra.py` (m.in. skumulowane AR, forest plot, heatmapa korelacji, okno zdarzenia, rolling vol) → `figures/extra_*.pdf`.

---

## Uruchomienie

Z katalogu głównego repozytorium:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python3 src/data_fetch.py              # zapis: data_cache/prices.parquet (wymaga sieci)
# Upewnij się, że masz data_cache/metadata_tickers.csv (w repo jest wzorcowy plik).

python3 src/task2_four_groups.py       # Zad 2
python3 src/task1_two_portfolios.py    # Zad 1
python3 src/task3_paired_event.py      # Zad 3
python3 src/plots_extra.py             # opcjonalnie

cd report && latexmk -pdf raport.tex
```

---

## Przykładowy output (uruchomienie na zapisanym cache)

Poniżej skrót komunikatów z konsoli przy aktualnym `data_cache/prices.parquet` (wartości mogą się zmienić po ponownym pobraniu danych).

**Zad 1** — po 2022-11-30: AI infra *n*=42, średnia mies. AR ≈ **0,026**; Defensive *n*=42, średnia ≈ **−0,011**. Levene: wariancje **nie** homogeniczne (*p*≈0,0005). **Welch**: *t*≈2,74, *p*≈0,0079 → H₀ równości średnich **odrzucona**; różnica średnich z 95% CI ≈ **[0,010; 0,064]**; Hedges *g*≈0,59; bootstrap CI dla różnicy średnich zawiera dodatnie wartości; Mann-Whitney *p*≈0,032. Wykres: `figures/task1_distributions.pdf`.

**Zad 2** — średnie per grupa (m.in. Sharpe): AI_infra wyraźnie wyższy Sharpe niż pozostałe trzy grupy; **ANOVA** dla Sharpe *F*≈4,96, *p*≈0,0054 → H₀ **odrzucona**; Tukey: różnice **AI_infra** vs każda z pozostałych grup istotne (*p*<0,05). Dla **alfy CAPM** również ANOVA istotna (*p*≈0,00043); Tukey: **AI_infra** vs **AI_platforms** i vs **Broad_tech** istotne, vs Defensive — nie. Wykresy: `figures/task2_sharpe_per_group.pdf`, `figures/task2_alpha_per_group.pdf`.

**Zad 3** — 21 spółek; dla zdarzenia 2022-11-30 przy oknie **120** sesji: średnia różnica (średnia AR po − przed) ≈ **+0,00243**, 95% CI dodatnie, główny test **Wilcoxon** *p*≈8,4×10⁻⁵ → H₀ braku różnicy **odrzucona**. Placebo 2021-11-30 przy tych oknach daje ujemne średnie różnic z istotnymi *p* (interpretacja: wzorzec specyficzny dla daty boomu, nie ogólny „szum”). Wykres: `figures/task3_sensitivity_placebo.pdf`. CSV: m.in. `data_cache/task3_sensitivity.csv`, `data_cache/task3_pairs_w120.csv`.

---

## Artefakty

| Lokalizacja | Zawartość |
|-------------|-----------|
| `data_cache/prices.parquet` | Ceny zamknięcia (po `data_fetch.py`) |
| `data_cache/metadata_tickers.csv` | Ticker, grupa, uzasadnienie |
| `data_cache/task1_*.csv` | Wyniki testów i szereg AR (Zad 1) |
| `data_cache/task2_*.csv` | Metryki i testy normalności (Zad 2) |
| `data_cache/task3_*.csv` | Wrażliwość na długość okna + pary (Zad 3) |
| `figures/` | PDF-y wykresów używanych w raporcie i opcjonalnie `extra_*.pdf` |

---

## Uwagi techniczne

- Skrypty zakładają uruchomienie z **korzenia projektu** (ścieżki względem `data_cache/` i `figures/`).
- `data_fetch.py` zapisuje **`prices.parquet`**; **`metadata_tickers.csv`** należy utrzymać zgodny z listą tickerów w `GROUPS` w `data_fetch.py` (w repozytorium jest gotowy plik).
- Do PDF z LaTeX potrzebny jest m.in. **latexmk** (np. TeX Live / MacTeX).
