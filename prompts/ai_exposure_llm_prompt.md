# Prompt do walidacji AI Exposure przez Claude Haiku

**Version**: 1.0 (utworzony 2026-05-20)
**Model docelowy**: `claude-haiku-4-5` lub aktualny stable Haiku
**Temperature**: 0
**Max tokens**: 200

---

## System prompt

```
Jesteś analitykiem finansowym specjalizującym się w analizie raportów rocznych 10-K
spółek publicznych. Twoim zadaniem jest ocena ekspozycji spółki na sztuczną
inteligencję (AI) i pokrewne technologie (machine learning, deep learning, neural
networks) na podstawie wyciągów z raportu 10-K spółki.

Oceniaj WYŁĄCZNIE na podstawie dostarczonego tekstu. Nie używaj wiedzy o spółce
spoza dostarczonych wycinków. Nie sugeruj się tym, co wiesz o spółce z roku 2023
lub późniejszych — raport 10-K został złożony PRZED listopadem 2022.

Skala oceny:
- 0 = brak ekspozycji na AI
- 1 = marginalna wzmianka (AI wymieniona, ale nie ma istotnego znaczenia dla działalności)
- 2 = istotny element działalności (AI jest jednym z produktów lub procesów spółki)
- 3 = core business / strategiczna ekspozycja (AI jest centralna dla modelu biznesowego)

Format odpowiedzi (DOKŁADNIE w tej strukturze):
SCORE: <0|1|2|3>
REASONING: <jedno zdanie po polsku uzasadniające>
```

---

## User prompt template

```
Ticker: {ticker}
Spółka: {company_name}
Filing date: {filing_date}
Period of report: {period_of_report}

Poniżej znajdują się wycinki z sekcji raportu 10-K spółki (Item 1: Business,
Item 1A: Risk Factors, Item 7: MD&A). Wycinki zawierają fragmenty zawierające
terminy związane z AI lub kontekst dla terminu "GPU".

Wycinki:
---
{excerpts}
---

Oceń ekspozycję tej spółki na AI w skali 0-3 i podaj jednoznaczne uzasadnienie.
```

---

## Pre-processing tekstów

1. Z surowego 10-K wybieramy sekcje Item 1, 1A, 7.
2. Wyodrębniamy zdania zawierające termin ze słownika konserwatywnego LUB termin "GPU" z kontekstem AI.
3. Łączymy w jeden tekst z separatorami `...` między fragmentami.
4. Truncate do maks. ~12 000 tokenów (Claude Haiku context window ≈ 200K, ale chcemy oszczędzać koszty).

## Parsing odpowiedzi

Regex: `SCORE:\s*([0-3])\s*\nREASONING:\s*(.+)`

Jeśli parsing nie powiedzie się → score = NaN, log do `llm_raw_responses.jsonl` z flagą `parse_error=true`.

## Mapowanie do `ai_exposure_llm` (znormalizowane do [0,1])

```python
score_to_normalized = {0: 0.0, 1: 0.33, 2: 0.67, 3: 1.0}
ai_exposure_llm = score_to_normalized[parsed_score]
```

## Zapisywane pola w `llm_raw_responses.jsonl`

```json
{
  "ticker": "NVDA",
  "model_name": "claude-haiku-4-5",
  "model_version": "claude-haiku-4-5-20251001",
  "temperature": 0,
  "max_tokens": 200,
  "prompt_sha256": "abc123...",
  "raw_response": "SCORE: 3\nREASONING: ...",
  "parsed_score": 3,
  "ai_exposure_llm": 1.0,
  "parse_error": false,
  "timestamp": "2026-05-20T20:55:00Z",
  "filing_date": "2022-02-18",
  "excerpts_char_count": 8421
}
```
