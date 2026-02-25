from __future__ import annotations

import os
import requests


def send_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, json={
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True
    }, timeout=20)
    resp.raise_for_status()


def maybe_send_from_env(env_token: str, env_chat_id: str, text: str) -> bool:
    token = os.getenv(env_token, "")
    chat_id = os.getenv(env_chat_id, "")
    if not token or not chat_id:
        return False
    send_telegram(token, chat_id, text)
    return True
