# Decision log

Chronologiczny rejestr decyzji analitycznych. Każdy wpis: data, decyzja, alternatywy, uzasadnienie.

---

## 2026-05-20 — Pre-rejestracja
Spisano `docs/PRE_REGISTRATION.md` zawierający hipotezy H1-H4, główną specyfikację Y, X, kontroli, modeli M1-M4 oraz robustness checks. Pre-rejestracja powstała PRZED uruchomieniem centralnej cross-sectional regression (czyli przed Etapem 3 planu).

## 2026-05-20 — Wybór percentile rank zamiast min-max scaling dla AI Exposure
**Decyzja**: `ai_exposure_det_main = percentile_rank(log_mentions_per_10k_words)`.
**Alternatywy rozważone**:
- min-max scaling (odrzucone — wrażliwe na outliery typu NVIDIA),
- z-score (odrzucone jako main — asymetryczny rozkład mentions, ale zostawione jako robustness).
**Uzasadnienie**: percentile rank jest odporny na obserwacje skrajne i interpretowalny ("w którym kwantylu ekspozycji AI jest spółka X").

## 2026-05-20 — GPU jako term context-conditional
**Decyzja**: `GPU` liczone TYLKO gdy w oknie ±50 słów występuje `AI / machine learning / training / inference / data center / accelerator / neural network / deep learning`.
**Uzasadnienie**: bez kontekstu `GPU` może oznaczać gaming chips bez związku z AI (np. AMD/NVDA w segmencie konsumenckim). Dodanie kontekstu zwiększa precyzję pomiaru bez wprowadzania post-2022 hype terms.

## 2026-05-20 — Conservative dictionary jako main, extended jako robustness
**Decyzja**: główna miara liczona ze słownika konserwatywnego (12 terminów istniejących w finansowym dyskursie pre-2022). Słownik rozszerzony (`generative AI, large language model, foundation model, transformer model, inference, training compute, AI accelerator`) tylko w robustness.
**Uzasadnienie**: ochrona przed *semantic look-ahead bias* — terminy typu "generative AI" lub "LLM" stały się popularne dopiero po premierze ChatGPT, więc obecność tych terminów w 10-K za FY2021 mogłaby świadczyć o nietypowo zaawansowanej firmie (selection bias).

## 2026-05-20 — LLM jako walidacja, nie składowa main scoru
**Decyzja**: główne wyniki raportujemy osobno dla `det_main` (specyfikacja główna) i osobno dla `llm` (robustness). NIE używamy arbitralnego mixu typu `0.6*det + 0.4*llm`.
**Alternatywy rozważone**: mix 60/40 (odrzucone), tylko LLM (odrzucone — czarna skrzynka), tylko deterministic (odrzucone — straciłoby walidację jakościową).
**Uzasadnienie**: mix arbitralny sterowałby wynikiem przez wagę. Osobne raportowanie pokazuje stabilność wniosku w niezależnych miarach.

## 2026-05-20 — n=41 → parsymonia w cross-sectional regression
**Decyzja**: M1-M4 z maks. 5 kontrolami. NIE wprowadzamy 11 GICS sector dummies ani interakcji.
**Uzasadnienie**: przy n=41 i 5 zmiennych już mamy ~7 obserwacji na zmienną; dodanie dummies obciążyłoby model i zredukowało stopnie swobody.

## 2026-05-20 — Wybór SPY jako benchmark (nie QQQ)
**Decyzja**: benchmark = SPY.
**Alternatywy rozważone**: QQQ (NASDAQ-100, bardziej tech-tilted), VTI (broad market).
**Uzasadnienie**: SPY to standard finansowy ("the market"). QQQ jako benchmark dla tech jest *circular* — wszystkie analizowane spółki to (głównie) tech, więc QQQ podcinałby premium AI z definicji.

## 2026-05-20 — FF RF dla modeli czynnikowych, ^IRX dla Sharpe Ratio
**Decyzja**: w modelach czynnikowych (CAPM/FF3/Carhart) używamy `RF` z serii Fama-French. W metrykach z Zadania 2 (Sharpe, Sortino) zostaje `^IRX`.
**Uzasadnienie**: mieszanie różnych proxies RF byłoby niespójne. Dla modeli czynnikowych RF z French Data Library jest standardem (matched do tych samych okien czasowych co czynniki SMB/HML/MOM).
