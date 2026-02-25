# krx-signal-forge-fdr

KR universe + swing scan using **FinanceDataReader** (PyKRX-free).

## Why
KRX data endpoints moved toward login/membership flows, which frequently breaks scrapers (PyKRX). citeturn0search0  
So this repo uses FinanceDataReader as the data adapter. citeturn0search1

> Note: FinanceDataReader also depends on upstream sources (KRX/NAVER/etc.), so we keep **cache + rollback + validation** for universe stability. citeturn0search6turn0search10

## What it does
- Universe build (KOSPI+KOSDAQ):
  - market cap Top 1200
  - price floor
  - liquidity prefilter using listing `Amount` (if available)
  - backups + rollback if build output is too small
- Daily scan (G1/G2/G3):
  - G1 uses **20D turnover (approx = close*volume)** from fetched OHLCV
  - G2 trend + EMA reclaim setup
  - G3 volume spike + breakout trigger
- Suppresses duplicate HARD alerts via `state/last_signals.json`

## Quickstart
```bash
pip install -r requirements.txt
python scripts/update_universe.py
python -m src.runner
```

## Config
- `config/universe_rules.json`
- `config/scan_rules.json`

## GitHub Actions
- `Update KRX Universe (FDR)` : KST 07:10
- `Daily KRX Scan (FDR)` : KST 16:20
