# AI Boom Premium? Pre-event AI Exposure, Model Risk and Validated Investment Analytics

**Working paper** | Roksana Kulawiec · Sylwia Malinowska · Mateusz Konopka · Antoni Grabowski
**Metody Badań w Finansach — ćwiczenia 3** | maj 2026
**Repozytorium**: https://github.com/Smalou/metody_badania_w_finansach

---

## Streszczenie

Projekt sprawdza, czy spółki ujawniające większą ekspozycję na AI w raportach
10-K SEC *przed* premierą ChatGPT (30.11.2022) osiągnęły wyższe post-event
abnormal returns. Główna innowacja to **pre-event AI Exposure Score** zbudowany
z dokumentów dostępnych publicznie przed datą zdarzenia, co częściowo ogranicza
look-ahead bias w klasyfikacji spółek jako „AI-related”.

**Główny wynik**: w prostej cross-sectional regression bez kontroli ryzyka
AI exposure wyjaśnia post-event AR (β₁=+0.26, p=0.044), ale po dodaniu pre-event
bety i volatility efekt **całkowicie zanika** (β₁≈0 we wszystkich modelach M2–M4).
Modele czynnikowe (CAPM/FF3/Carhart 4) potwierdzają: AI infrastructure portfolio
ma dodatnią szacowaną alfę (+8.7% do +14.7% rocznie) ale nieistotną na poziomie 0.10
przy HAC SE. Praktyczna implikacja: „AI Premium” okresu 2022–2026 to w znacznej
mierze premium za high-beta tilt, a nie czysty efekt fundamentalny.

---

## Struktura repozytorium

```
cwiczenia_3/
├── src/                      # Skrypty analityczne
│   ├── data_fetch.py         # yfinance: 43 instrumenty → prices.parquet
│   ├── sec_edgar_fetcher.py  # SEC EDGAR: 41 raportów 10-K (filing_date<2022-11-30)
│   ├── nlp_ai_exposure.py    # Deterministic AI Exposure Score (główna miara)
│   ├── llm_classifier.py     # Claude Haiku validation (--no-llm fallback)
│   ├── cross_sectional.py    # Główna regresja M1–M4 + bootstrap + White SE
│   ├── robustness.py         # LOO, exclude firms, alt scores/windows/scaling
│   ├── factor_models.py      # CAPM / FF3 / Carhart 4 z Newey-West HAC SE
│   ├── task1_two_portfolios.py  # AI infra vs Defensive (Welch + bootstrap)
│   ├── task2_four_groups.py     # ANOVA + Tukey HSD na Sharpe i alpha
│   ├── task3_paired_event.py    # Paired A/B + placebo + CAPM-adjusted
│   ├── plots_extra.py        # Wykresy diagnostyczne
│   ├── common.py             # log_returns, EW portfolios, abnormal returns
│   ├── metrics.py            # 7 metryk: Sharpe/Sortino/MaxDD/beta/alpha
│   └── plot_utils.py         # styl + savefig
├── data_cache/               # parquet + csv z wynikami
├── figures/                  # wszystkie wykresy PDF
├── report/
│   ├── raport.tex            # ŹRÓDŁO 34-stronicowego working paper
│   ├── raport.pdf            # FINALNY DELIVERABLE
│   └── refs.bib              # bibliografia BibTeX
├── docs/
│   ├── PRE_REGISTRATION.md   # hipotezy + model + robustness — spisane PRZED regresją
│   ├── decision_log.md       # chronologiczny rejestr decyzji
│   └── DATA_LICENSES.md      # źródła i licencje
├── prompts/
│   └── ai_exposure_llm_prompt.md  # prompt v1.0 (SHA-256 logowany)
├── src_ascii/                # ASCII kopie src/ dla LaTeX listings
├── materialy/                # Treść zadań kursowych (PL)
├── Makefile                  # pełna reprodukcja: make all
├── requirements.txt          # wymagane pakiety (luźne wersje)
├── requirements-lock.txt     # pip freeze (pinned wersje)
├── CITATION.cff              # cytowanie
├── LICENSE-CODE              # MIT
└── LICENSE-DOCS              # CC BY 4.0
```

---

## Replikacja

### Wymagania
- Python 3.13+
- LaTeX z pakietem `latexmk` (do budowy PDF)
- Opcjonalnie: `ANTHROPIC_API_KEY` w env (Claude Haiku) — bez klucza pipeline
  działa nadal z deterministycznym scoringiem jako specyfikacją główną.

### Setup
```bash
git clone https://github.com/Smalou/metody_badania_w_finansach
cd metody_badania_w_finansach/cwiczenia_3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

### Pełna reprodukcja
```bash
make all          # data → SEC → NLP → analizy → robustness → factors → raport
```

### Selektywne uruchomienie
```bash
make data         # tylko pobranie cen z yfinance
make sec          # tylko pobranie 10-K z SEC EDGAR
make nlp          # tylko AI Exposure Score
make analysis     # Zad 1/2/3 + cross-sectional regression
make robustness   # leave-one-out + exclude firms + alt specifications
make factors      # CAPM / FF3 / Carhart 4
make report       # kompilacja PDF
```

### Standalone kompilacja PDF (bez Makefile)

Raport zakłada strukturę repozytorium z folderem `figures/` jeden poziom wyżej
względem `report/`. Aby skompilować PDF z surowego `.tex`:

```bash
cd report
latexmk -pdf raport.tex
```

Folder `report/` zawiera `raport.tex` oraz `refs.bib`. Pliki `figures/*.pdf`
i `src_ascii/*.py` są ładowane przez ścieżki względne `../figures/` i `../src_ascii/`.

---

## Źródła danych

| Źródło | Co używamy | Licencja |
|---|---|---|
| Yahoo Finance via `yfinance` | Dzienne ceny 43 instrumentów (41 spółek + SPY + ^IRX), 2021-01-04 → 2026-05-14 | Yahoo Terms of Use (akademickie OK) |
| SEC EDGAR | 41 raportów 10-K złożonych przed 2022-11-30 | Public domain (17 U.S.C. § 105) |
| Kenneth French Data Library | Dzienne czynniki Fama-French 3 + Momentum (1926-2026) | Free academic use |
| Anthropic Claude Haiku 4.5 | LLM validation AI Exposure (opcjonalne) | Anthropic Usage Policies |

Szczegóły: `docs/DATA_LICENSES.md`.

---

## Kluczowe wyniki

### Zadanie 2 (najmocniejszy klasyczny wynik)
- **Sharpe Ratio**: ANOVA *F(3,37)*=4.96, *p*=0.005, ω²=0.23. Tukey: AI infrastructure
  istotnie wyższe od każdej z 3 pozostałych grup.
- **CAPM alpha**: ANOVA *F(3,37)*=7.62, *p*=4·10⁻⁴, ω²=0.33. AI infrastructure
  istotnie wyższe niż AI platforms i Broad tech; różnica vs Defensive nieistotna.

### Cross-sectional regression (M1–M4)
| Model | β₁ (AI_Exp) | p_HC1 | 95% CI HC1 |
|---|---|---|---|
| M1 | **+0.259*** | 0.044 | [+0.007; +0.512] |
| M2 (+ β + vol) | -0.009 | 0.93 | [-0.221; +0.203] |
| M3 (+ momentum) | +0.013 | 0.91 | [-0.206; +0.232] |
| M4 (+ log mc) | +0.044 | 0.68 | [-0.169; +0.257] |

### Robustness
- **Leave-one-out**: 7/41 sign changes (17%), najbardziej wpływowa IBM.
- **Bez NVIDIA**: β₁ z M3 odwraca znak (z +0.013 na −0.036).
- **12-miesięczne okno**: β₁=+0.254, *p*=0.031 (jedyne istotne odstępstwo).

### Factor models (annualized alpha, HAC SE)
| Grupa | CAPM α | FF3 α | Carhart 4 α |
|---|---|---|---|
| **AI infrastructure** | +8.7% (p=0.41) | +14.7% (p=0.15) | +13.2% (p=0.19) |
| AI platforms | -9.3% (p=0.13) | -4.4% (p=0.39) | -3.9% (p=0.44) |
| Broad tech | -8.9% (p=0.05) | -6.5% (p=0.14) | -6.2% (p=0.16) |
| Defensive | +2.6% (p=0.62) | -0.8% (p=0.87) | -0.3% (p=0.95) |

**Wniosek**: AI Premium *istnieje* w surowych zwrotach, ale *ginie* pod kontrolą
ryzyka systematycznego.

---

## Cytowanie

```bibtex
@misc{KMK2026AIPremium,
  author = {Kulawiec, Roksana and Malinowska, Sylwia and Konopka, Mateusz and Grabowski, Antoni},
  title  = {AI Boom Premium? Pre-event AI Exposure, Model Risk and Validated Investment Analytics},
  year   = {2026},
  note   = {Working paper, Metody Badań w Finansach},
  url    = {https://github.com/Smalou/metody_badania_w_finansach}
}
```

---

## Licencje

- **Kod** (`src/`, `Makefile`, `*.py`): MIT — `LICENSE-CODE`
- **Dokumentacja i raport** (`report/`, `docs/`, `README.md`): CC BY 4.0 — `LICENSE-DOCS`
