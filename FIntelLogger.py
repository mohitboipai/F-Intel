"""
FIntelLogger.py — Structured JSONL logging module for the F-Intel platform.

Provides:
  - get_logger(source_name)          : Returns a logger writing JSONL to dated/timestamped files.
  - stream_to_log(process, source)   : Reads subprocess stdout, logs lines, extracts metrics.
  - cleanup_old_logs(max_days)       : Purges log date-folders older than max_days.
  - get_session_summary()            : Returns session stats dict.

Uses only Python standard library. Zero imports from the F-Intel codebase.
"""

import logging
import logging.handlers
import pathlib
import json
import datetime
import re
import threading
import os

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE  (module-level, shared across all loggers in one process)
# ──────────────────────────────────────────────────────────────────────────────

_SESSION_START: datetime.datetime = datetime.datetime.now()
_SESSION_START_ISO: str = _SESSION_START.isoformat()

# Datestamp and time-stamp strings used for all file names in this session
_DATE_STR: str = _SESSION_START.strftime("%Y-%m-%d")
_TIME_STR: str = _SESSION_START.strftime("%H-%M-%S")

# Project root = directory that contains this file
_PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).parent.resolve()
_LOG_BASE: pathlib.Path = _PROJECT_ROOT / "logs"
_SESSION_DIR: pathlib.Path = _LOG_BASE / _DATE_STR

# Per-source line counters and metrics counter — protected by a lock
_LOCK = threading.Lock()
_LINE_COUNTS: dict[str, int] = {"Launcher": 0, "DataServer": 0, "Tunnel": 0, "Analyzer": 0}
_METRICS_COUNT: int = 0

# Registry: source_name → (Logger, metrics_file_path, current_part_index)
_LOGGERS: dict[str, logging.Logger] = {}
_METRICS_PATHS: dict[str, pathlib.Path] = {}

# Maximum log file size before rotation (10 MB)
_MAX_LOG_BYTES: int = 10 * 1024 * 1024


# ──────────────────────────────────────────────────────────────────────────────
# JSONL HANDLER — writes one JSON object per log record
# ──────────────────────────────────────────────────────────────────────────────

class _JsonlHandler(logging.StreamHandler):
    """Writes structured JSONL records.  Handles 10 MB per-file rotation."""

    def __init__(self, base_path: pathlib.Path, source: str) -> None:
        self._base_path = base_path  # e.g. logs/2026-01-01/dataserver_10-00-00
        self._source = source
        self._part = 1
        self._file_obj = self._open_file()
        super().__init__(self._file_obj)

    # ── file management ───────────────────────────────────────────────────────

    def _current_path(self) -> pathlib.Path:
        if self._part == 1:
            return pathlib.Path(str(self._base_path) + ".log")
        return pathlib.Path(str(self._base_path) + f"_part{self._part}.log")

    def _open_file(self):
        path = self._current_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, "a", encoding="utf-8", buffering=1)

    def _rotate_if_needed(self) -> None:
        try:
            if self._file_obj and os.path.getsize(self._current_path()) >= _MAX_LOG_BYTES:
                self._file_obj.flush()
                self._file_obj.close()
                self._part += 1
                self._file_obj = self._open_file()
                self.stream = self._file_obj
        except (OSError, AttributeError):
            pass

    # ── emit ──────────────────────────────────────────────────────────────────

    def emit(self, record: logging.LogRecord) -> None:
        self._rotate_if_needed()
        try:
            level_map = {
                logging.DEBUG:    "DEBUG",
                logging.INFO:     "INFO",
                logging.WARNING:  "WARN",
                logging.ERROR:    "ERROR",
                logging.CRITICAL: "CRITICAL",
            }
            level = level_map.get(record.levelno, "INFO")
            obj = {
                "ts":     datetime.datetime.now().isoformat(),
                "level":  level,
                "source": self._source,
                "msg":    record.getMessage(),
            }
            self._file_obj.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._file_obj.flush()
        except Exception:
            self.handleError(record)


# ──────────────────────────────────────────────────────────────────────────────
# METRICS REGEX PATTERNS
# ──────────────────────────────────────────────────────────────────────────────

# DataServer metrics
_DS_SPOT_RE    = re.compile(r'"spot"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
_DS_LIVE_RE    = re.compile(r'Live', re.IGNORECASE)

# VolatilityAnalyzer metrics
_VA_REGIME_RE  = re.compile(r'Regime:\s*(\w+)')
_VA_VRP_RE     = re.compile(r'VRP:\s*([+-]?[0-9]+(?:\.[0-9]+)?)')
_VA_COMP_RE    = re.compile(r'COMPOSITE:\s*([+-]?[0-9]+(?:\.[0-9]+)?)')
_VA_ACTION_RE  = re.compile(r'ACTION:\s*(\w+)')
_VA_ATMIV_RE   = re.compile(r'ATM IV:\s*([+-]?[0-9]+(?:\.[0-9]+)?)')


def _extract_metrics(line: str, source: str) -> list[dict]:
    """Return a list of metric dicts extracted from *line* for the given source."""
    results = []
    ts = datetime.datetime.now().isoformat()

    if source == "DataServer":
        m = _DS_SPOT_RE.search(line)
        if m:
            results.append({"ts": ts, "source": source, "metric": "spot",
                             "value": float(m.group(1))})
        if _DS_LIVE_RE.search(line):
            results.append({"ts": ts, "source": source, "metric": "chain_refresh", "value": 1})

    elif source == "Analyzer":
        m = _VA_REGIME_RE.search(line)
        if m:
            results.append({"ts": ts, "source": source, "metric": "regime",
                             "value": m.group(1)})
        m = _VA_VRP_RE.search(line)
        if m:
            results.append({"ts": ts, "source": source, "metric": "vrp",
                             "value": float(m.group(1))})
        m = _VA_COMP_RE.search(line)
        if m:
            results.append({"ts": ts, "source": source, "metric": "composite_score",
                             "value": float(m.group(1))})
        m = _VA_ACTION_RE.search(line)
        if m:
            results.append({"ts": ts, "source": source, "metric": "action",
                             "value": m.group(1)})
        m = _VA_ATMIV_RE.search(line)
        if m:
            results.append({"ts": ts, "source": source, "metric": "atm_iv",
                             "value": float(m.group(1))})

    return results


# ──────────────────────────────────────────────────────────────────────────────
# TUNNEL URL REGEX PATTERNS (used in stream_to_log when source == "Tunnel")
# ──────────────────────────────────────────────────────────────────────────────

_TUNNEL_P1 = re.compile(r'https://[a-z0-9\-]+\.trycloudflare\.com')
_TUNNEL_P2 = re.compile(r'https://[a-z0-9\-]+\.cfargotunnel\.com')
_TUNNEL_P3 = re.compile(r'https://[^\s]+')
_KNOWN_TLDS = (".com", ".net", ".org", ".io", ".dev", ".app")


def _extract_tunnel_url(line: str) -> str | None:
    """Return the first tunnel URL found in *line*, or None."""
    m = _TUNNEL_P1.search(line)
    if m:
        return m.group(0)
    m = _TUNNEL_P2.search(line)
    if m:
        return m.group(0)
    # Pattern 3 — generic https fallback with TLD validation
    m = _TUNNEL_P3.search(line)
    if m:
        url = m.group(0)
        if "." in url and any(url.endswith(tld) or (tld + "/") in url or (tld + "?") in url
                              for tld in _KNOWN_TLDS):
            return url
    return None


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def get_logger(source_name: str) -> logging.Logger:
    """Return a configured JSONL logger for *source_name*.

    source_name must be one of: "Launcher", "DataServer", "Tunnel", "Analyzer"
    Thread-safe. Handles 10 MB rotation internally via _JsonlHandler.
    """
    with _LOCK:
        if source_name in _LOGGERS:
            return _LOGGERS[source_name]

        # Determine file base path
        prefix_map = {
            "Launcher":   "launcher",
            "DataServer": "dataserver",
            "Tunnel":     "tunnel",
            "Analyzer":   "analyzer",
        }
        prefix = prefix_map.get(source_name, source_name.lower())
        _SESSION_DIR.mkdir(parents=True, exist_ok=True)
        base_path = _SESSION_DIR / f"{prefix}_{_TIME_STR}"

        # Build logger
        logger = logging.getLogger(f"fintel.{source_name}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        handler = _JsonlHandler(base_path, source_name)
        logger.addHandler(handler)

        _LOGGERS[source_name] = logger

        # Metrics file path (used by stream_to_log)
        _METRICS_PATHS[source_name] = _SESSION_DIR / f"metrics_{_TIME_STR}.jsonl"

        return logger


def stream_to_log(process, source_name: str, url_callback=None) -> None:
    """Read *process*.stdout line by line and write to the source logger.

    Extracts metrics and writes them to the metrics JSONL file.
    If *url_callback* is not None and *source_name* is "Tunnel", applies tunnel URL
    regex to each line and calls url_callback(url) exactly once on first match.

    Intended to run in a daemon thread.
    """
    global _METRICS_COUNT

    logger = get_logger(source_name)
    metrics_path = _METRICS_PATHS.get(source_name, _SESSION_DIR / f"metrics_{_TIME_STR}.jsonl")

    # For the Tunnel source: ensure the callback fires only once
    _url_fired = threading.Event()

    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n\r")

            # ── Log the raw line ──────────────────────────────────────────────
            logger.info(line)
            with _LOCK:
                if source_name in _LINE_COUNTS:
                    _LINE_COUNTS[source_name] += 1

            # ── Extract and persist metrics ───────────────────────────────────
            metrics = _extract_metrics(line, source_name)
            if metrics:
                try:
                    metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(metrics_path, "a", encoding="utf-8", buffering=1) as mf:
                        for m in metrics:
                            mf.write(json.dumps(m, ensure_ascii=False) + "\n")
                    with _LOCK:
                        _METRICS_COUNT += len(metrics)
                except OSError:
                    pass

            # ── Tunnel URL detection ──────────────────────────────────────────
            if source_name == "Tunnel" and url_callback is not None:
                if not _url_fired.is_set():
                    url = _extract_tunnel_url(line)
                    if url:
                        _url_fired.set()
                        try:
                            url_callback(url)
                        except Exception:
                            pass

    except (OSError, ValueError):
        # stdout closed or process ended
        pass


def cleanup_old_logs(max_days: int = 7) -> None:
    """Delete date folders under logs/ that are older than *max_days* days."""
    _LOG_BASE.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.date.today() - datetime.timedelta(days=max_days)

    for entry in _LOG_BASE.iterdir():
        if not entry.is_dir():
            continue
        # Expect folder names in YYYY-MM-DD format
        try:
            folder_date = datetime.date.fromisoformat(entry.name)
        except ValueError:
            continue
        if folder_date < cutoff:
            try:
                import shutil
                shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                pass


def get_session_summary() -> dict:
    """Return a summary dict of this session's logging activity."""
    with _LOCK:
        return {
            "session_start":   _SESSION_START_ISO,
            "lines_logged":    dict(_LINE_COUNTS),
            "metrics_captured": _METRICS_COUNT,
            "log_dir":         str(_SESSION_DIR),
        }


# ──────────────────────────────────────────────────────────────────────────────
# MODULE INITIALISATION — run cleanup on import
# ──────────────────────────────────────────────────────────────────────────────
cleanup_old_logs(max_days=7)
