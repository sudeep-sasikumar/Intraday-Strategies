"""Telegram messages for paper-trade signals. Without TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
in the environment, messages are only printed."""
from __future__ import annotations

import os

import httpx


def send(text: str) -> bool:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
    if not (token and chat):
        print(f"[telegram not configured] {text}", flush=True)
        return False
    try:
        r = httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                       json={"chat_id": chat, "text": text[:4096], "link_preview_options": {"is_disabled": True}}, timeout=20)
        return r.status_code == 200
    except httpx.HTTPError as e:
        print(f"[telegram failed: {e}] {text}", flush=True)
        return False
