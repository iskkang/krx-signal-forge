from __future__ import annotations
import os
import requests


def send_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("[Telegram] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        r = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
                # "parse_mode": "Markdown"  # 필요하면 주석 해제
            },
            timeout=15,
        )
        if r.status_code != 200:
            print("[Telegram] HTTP error:", r.status_code, r.text)
    except Exception as e:
        print("[Telegram] Exception:", repr(e))
