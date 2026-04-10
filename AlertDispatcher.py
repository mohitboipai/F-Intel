# AlertDispatcher.py
# ──────────────────────────────────────────────────────────────────────────────
# Central alert bus for the F-Intel platform.
#
# Modules call the module-level `fire()` shortcut:
#   from AlertDispatcher import fire
#   fire("GammaExplosionModel", "CRITICAL", "title", "body", data={...})
#
# Every alert is:
#   1. Written to alerts.jsonl (always)
#   2. Printed to console (always)
#   3. Sent to Telegram (WARNING/CRITICAL only, if env vars are set)
#
# Env vars (optional):
#   FINTEL_TG_TOKEN  — Telegram bot token
#   FINTEL_TG_CHAT   — Telegram chat_id (integer)
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
import urllib.parse
from datetime import datetime

ALERT_LOG = "alerts.jsonl"


class AlertDispatcher:
    """
    Central alert bus. Modules call .fire() with an event dict.
    Writes to alerts.jsonl. Optional Telegram webhook if env var set.
    """

    def __init__(self):
        self.telegram_token = os.environ.get("FINTEL_TG_TOKEN", "")
        self.telegram_chat  = os.environ.get("FINTEL_TG_CHAT", "")

    def fire(self, source: str, level: str, title: str, body: str, data: dict = None):
        """
        source: module name e.g. 'GammaExplosionModel'
        level:  'INFO' | 'WARNING' | 'CRITICAL'
        title:  short one-line summary
        body:   detail string
        data:   optional dict of raw values
        """
        event = {
            "ts":     datetime.now().isoformat(),
            "source": source,
            "level":  level,
            "title":  title,
            "body":   body,
            "data":   data or {}
        }
        self._log(event)
        self._print(event)
        if self.telegram_token and level in ("WARNING", "CRITICAL"):
            self._send_telegram(event)

    def _log(self, event: dict):
        try:
            with open(ALERT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass  # never crash caller on log failure

    def _print(self, event: dict):
        icons = {"INFO": "[i]", "WARNING": "[!]", "CRITICAL": "[!!]"}
        icon = icons.get(event["level"], "[?]")
        print(f"\n{icon} ALERT [{event['source']}] {event['title']}")
        if event["body"]:
            print(f"    {event['body']}")

    def _send_telegram(self, event: dict):
        try:
            import urllib.request
            msg = f"{event['level']}: {event['title']}\n{event['body']}"
            url = (
                f"https://api.telegram.org/bot{self.telegram_token}"
                f"/sendMessage?chat_id={self.telegram_chat}"
                f"&text={urllib.parse.quote(msg)}"
            )
            urllib.request.urlopen(url, timeout=3)
        except Exception:
            pass  # Telegram is best-effort; never crash caller


# Module-level singleton — imported once per process
_dispatcher = AlertDispatcher()


def fire(source: str, level: str, title: str, body: str, data: dict = None):
    """Module-level shortcut so callers don't need to manage the singleton."""
    _dispatcher.fire(source, level, title, body, data)
