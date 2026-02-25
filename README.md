# krx-signal-forge

KOSPI+KOSDAQ practical universe + swing scan skeleton (GitHub Actions friendly).

## What this repo does
- Builds **KOSPI+KOSDAQ** universe using PyKRX:
  - market cap Top 1200
  - 20D average turnover filter (KRW)
  - price floor filter
  - backups + rollback guard against PyKRX/KRX hiccups
- Runs a daily scan with a **G1/G2/G3** gate structure:
  - G1: price + turnover + history length
  - G2: MA200 trend + EMA20 reclaim setup (+ EMA alignment)
  - G3: volume spike + N-day breakout trigger
- Saves `state/last_signals.json` to suppress duplicate HARD alerts.

## Quickstart (local)
```bash
pip install -r requirements.txt
python scripts/update_universe.py
python -m src.runner
```

## Config
- `config/universe_rules.json` : universe build rules
- `config/scan_rules.json` : scan rules + optional telegram

## Telegram (optional)
1) Set in `config/scan_rules.json`:
   - `"telegram": { "enabled": true, ... }`
2) Add GitHub Secrets:
   - `TELEGRAM_TOKEN`
   - `TELEGRAM_CHAT_ID`

## Notes about PyKRX reliability
- PyKRX can break when KRX changes pages/login flow.
- This repo protects you by:
  - always creating a backup first
  - validating minimum universe size
  - keeping previous `universe.txt` on failure
