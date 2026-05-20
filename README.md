# Projekt zaliczeniowy IAD — ćwiczenia 3

Trzy zadania statystyczne na danych z `yfinance`:

1. **Zad 1** — porównanie średnich dziennych log-zwrotów ETF-ów `XLF` (Financials) vs `XLK` (Technology); t-Studenta lub Manna-Whitneya.
2. **Zad 2** — porównanie średnich Sharpe Ratio dla 4 sektorów GICS (Financials, Energy, Communication, Tech); ANOVA + Tukey lub Kruskal-Wallis + Dunn.
3. **Zad 3** — paired event study wokół wybuchu wojny na Bliskim Wschodzie (28.02.2026); paired t-test lub Wilcoxon dla 15 spółek naftowych USA.

Deliverable: `report/raport.pdf` zbudowany z `report/raport.tex`.

## Uruchomienie

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 src/data_fetch.py           # pobranie + cache do data_cache/
python3 src/task2_sharpe_anova.py   # Zad 2
python3 src/task1_two_groups.py     # Zad 1
python3 src/task3_paired_event.py   # Zad 3

cd report && latexmk -pdf raport.tex
```
