# krx-signal-forge

KRX swing signal scanner (GitHub Actions friendly).  
- Universe source: FinanceDataReader (KRX listing)  
- Universe ranking: Top 1200 by **20D average traded value** (Close * Volume)  
- Strategy (default): EMA20 reclaim + volume confirmation  
- Output: Hard candidates (trade-ready) + Soft watchlist (setup forming)

## Quick start (local)

```bash
pip install -r requirements.txt
python scripts/update_universe.py
python -m src.runner
```

## GitHub Actions

- `daily_scan.yml`: runs scan after market close (KST) on weekdays.
- Universe update runs weekly (Mon) by default inside the workflow.

## Env vars (optional)

- `TG_TOKEN`, `TG_CHAT_ID` to enable Telegram alerts (set `config/settings.json` telegram.enabled = true)
