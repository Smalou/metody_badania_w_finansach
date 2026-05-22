.PHONY: all data sec nlp llm analysis robustness factors plots report ascii clean help

PY := .venv/bin/python
SRC := src
REPORT := report

help:
	@echo "Cele:"
	@echo "  make data        - pobranie cen z yfinance (cache do data_cache/)"
	@echo "  make sec         - pobranie 10-K z SEC EDGAR"
	@echo "  make nlp         - deterministic AI Exposure Score"
	@echo "  make llm         - LLM validation (Claude Haiku); brak ANTHROPIC_API_KEY -> NaN"
	@echo "  make analysis    - Zad 1/2/3 + cross-sectional regression"
	@echo "  make robustness  - LOO + exclude + alt scores/windows/scaling"
	@echo "  make factors     - CAPM/FF3/Carhart 4 factor models"
	@echo "  make plots       - dodatkowe wykresy diagnostyczne"
	@echo "  make ascii       - regeneracja src_ascii/ dla LaTeX listings"
	@echo "  make report      - kompilacja report/raport.pdf"
	@echo "  make all         - pelny pipeline od zera"
	@echo "  make clean       - usun wynikowe artefakty (zachowac cache)"

data:
	$(PY) $(SRC)/data_fetch.py

sec:
	$(PY) $(SRC)/sec_edgar_fetcher.py

nlp:
	$(PY) $(SRC)/nlp_ai_exposure.py

llm:
	$(PY) $(SRC)/llm_classifier.py

analysis:
	$(PY) $(SRC)/task1_two_portfolios.py
	$(PY) $(SRC)/task2_four_groups.py
	$(PY) $(SRC)/task3_paired_event.py
	$(PY) $(SRC)/cross_sectional.py

robustness:
	$(PY) $(SRC)/robustness.py

factors:
	$(PY) $(SRC)/factor_models.py

plots:
	$(PY) $(SRC)/plots_extra.py

ascii:
	$(PY) -c "import unicodedata; from pathlib import Path; \
	    to_ascii = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', \
	      s.replace('ł','l').replace('Ł','L').replace('—','-').replace('–','-').replace('„','\"').replace('Ś','S').replace('ś','s').replace('é','e').replace('→','->')) if not unicodedata.combining(c)); \
	    Path('src_ascii').mkdir(exist_ok=True); \
	    [Path('src_ascii', f.name).write_text(to_ascii(f.read_text(encoding='utf-8')), encoding='ascii') for f in Path('src').glob('*.py')]"

report: ascii
	cd $(REPORT) && latexmk -pdf -interaction=nonstopmode raport.tex
	@echo "PDF: $(REPORT)/raport.pdf"

all: data sec nlp llm analysis robustness factors plots ascii report
	@echo "Pelna reprodukcja zakonczona."

clean:
	rm -f $(REPORT)/*.aux $(REPORT)/*.log $(REPORT)/*.out $(REPORT)/*.toc \
	       $(REPORT)/*.fls $(REPORT)/*.fdb_latexmk
	@echo "Usunieto artefakty LaTeX. Cache (data_cache/, figures/) zachowany."
