#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gửi thông báo Telegram.

Cấu hình theo thứ tự ưu tiên:
  1. env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  2. file: $TELEGRAM_CONFIG (mặc định ~/.config/opencode/telegram.json)
     {"bot_token": "123:ABC", "chat_id": 123456789}

Dùng:
  python telegram.py "nội dung"
  python telegram.py --chat 123456789 "nội dung"
  python telegram.py --whoami          # tự dò chat_id từ getUpdates
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

CONFIG = Path(os.environ.get("TELEGRAM_CONFIG", "/root/.config/opencode/telegram.json"))


def load_config() -> tuple[str | None, str | None]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if CONFIG.exists():
        try:
            payload = json.loads(CONFIG.read_text(encoding="utf-8"))
            token = token or payload.get("bot_token")
            chat = chat or payload.get("chat_id")
        except Exception as error:  # noqa: BLE001
            print(f"config lỗi: {error}", file=sys.stderr)
    return token, (str(chat) if chat else None)


def api(token: str, method: str, payload: dict | None = None) -> dict:
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = urllib.parse.urlencode(payload or {}).encode()
    with urllib.request.urlopen(url, data=data, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def discover_chat(token: str) -> str | None:
    result = api(token, "getUpdates")
    for update in reversed(result.get("result", [])):
        for key in ("message", "channel_post", "edited_message", "my_chat_member"):
            chat = (update.get(key) or {}).get("chat") or {}
            if chat.get("id") is not None:
                return str(chat["id"])
    return None


def send(text: str, token: str | None = None, chat: str | None = None) -> bool:
    token = token or load_config()[0]
    chat = chat or load_config()[1]
    if not token:
        print("thiếu TELEGRAM_BOT_TOKEN", file=sys.stderr)
        return False
    if not chat:
        chat = discover_chat(token)
        if not chat:
            print(
                "thiếu chat_id và không dò được. Hãy nhắn 1 tin cho bot rồi chạy lại.",
                file=sys.stderr,
            )
            return False
    try:
        result = api(token, "sendMessage", {
            "chat_id": chat,
            "text": text[:4000],
            "disable_web_page_preview": "true",
        })
        return bool(result.get("ok"))
    except Exception as error:  # noqa: BLE001
        print(f"gửi lỗi: {type(error).__name__}: {error}", file=sys.stderr)
        return False


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    if argv[1] == "--whoami":
        token, _ = load_config()
        if not token:
            print("thiếu TELEGRAM_BOT_TOKEN", file=sys.stderr)
            return 1
        chat = discover_chat(token)
        print(f"chat_id={chat}")
        return 0 if chat else 1
    if argv[1] == "--chat":
        chat = argv[2]
        text = " ".join(argv[3:])
        return 0 if send(text, chat=chat) else 1
    return 0 if send(" ".join(argv[1:])) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
