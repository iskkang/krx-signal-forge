from __future__ import annotations
import os
import requests

def send_message(text: str) -> None:
    token = os.getenv("TG_TOKEN", "").strip()
    chat_id = os.getenv("TG_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True}, timeout=15)
