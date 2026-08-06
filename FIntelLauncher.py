"""
FIntelLauncher.py — Single entry point for the F-Intel Options Intelligence Platform.

Replaces start_analyzer.ps1.

Launch order:
  1. Splash screen + logger init
  2. Expiry input → fintel_session.json
  3. DataServer.py   (piped, background daemon thread reads stdout)
  4. start_tunnel.py (piped, background daemon thread extracts URL)
  5. VolatilityAnalyzer.py (interactive — NOT piped)
  6. Process-monitor daemon loop
  7. Graceful shutdown + session summary

PyInstaller notes
-----------------
When packaged as a .exe via PyInstaller (--onefile), __file__ may point inside a
temporary extraction directory (_MEIPASS).  The *project root* (where DataServer.py
lives) is the directory that contains the .exe, i.e. the parent of sys.executable.
We detect this at runtime and set PROJECT_ROOT accordingly.
"""

import sys

# Ensure stdout can print Unicode (like ✓) on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    except AttributeError:
        pass
import os
import subprocess
import threading
import json
import datetime
import time
import re
import pathlib

import colorama
from colorama import Fore, Style

import FIntelLogger

# ──────────────────────────────────────────────────────────────────────────────
# COLORAMA INIT
# ──────────────────────────────────────────────────────────────────────────────
colorama.init(autoreset=True)

# ──────────────────────────────────────────────────────────────────────────────
# PROJECT ROOT RESOLUTION
# When running as a .py script: parent of this file.
# When running as a PyInstaller .exe (--onefile): parent of sys.executable,
# because __file__ points into _MEIPASS (temp extraction dir).
# ──────────────────────────────────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    # Packaged .exe — project root is the folder containing the .exe
    PROJECT_ROOT: pathlib.Path = pathlib.Path(sys.executable).parent.resolve()
else:
    PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

PYTHON_EXE: pathlib.Path = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS — coloured printing
# ──────────────────────────────────────────────────────────────────────────────

def _cyan(text: str) -> str:
    return f"{Style.BRIGHT}{Fore.CYAN}{text}{Style.RESET_ALL}"

def _yellow(text: str) -> str:
    return f"{Style.BRIGHT}{Fore.YELLOW}{text}{Style.RESET_ALL}"

def _green(text: str) -> str:
    return f"{Style.BRIGHT}{Fore.GREEN}{text}{Style.RESET_ALL}"

def _red(text: str) -> str:
    return f"{Style.BRIGHT}{Fore.RED}{text}{Style.RESET_ALL}"

def _white(text: str) -> str:
    return f"{Style.BRIGHT}{Fore.WHITE}{text}{Style.RESET_ALL}"

def _dim(text: str) -> str:
    return f"{Style.DIM}{Fore.WHITE}{text}{Style.RESET_ALL}"

def _bright_yellow(text: str) -> str:
    return f"{Style.BRIGHT}{Fore.YELLOW}{text}{Style.RESET_ALL}"


def _popen(script_name: str) -> subprocess.Popen:
    """Launch a script with piped stdout/stderr, UTF-8, line-buffered."""
    return subprocess.Popen(
        [str(PYTHON_EXE), "-u", script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _start_stream_thread(process: subprocess.Popen, source: str,
                          url_callback=None) -> threading.Thread:
    t = threading.Thread(
        target=FIntelLogger.stream_to_log,
        args=(process, source),
        kwargs={"url_callback": url_callback},
        daemon=True,
    )
    t.start()
    return t


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — SPLASH SCREEN
# ──────────────────────────────────────────────────────────────────────────────

def _step1_splash(session_ts: str):
    subprocess.run(["cls" if os.name == "nt" else "clear"], shell=True)

    banner = (
        "╔═══════════════════════════════════════════════════╗\n"
        "║          F-INTEL  ◆  OPTIONS INTELLIGENCE         ║\n"
        "║             NIFTY Real-Time Platform              ║\n"
        "╚═══════════════════════════════════════════════════╝"
    )
    print(_cyan(banner))
    print(_cyan(f"  Session: {session_ts}"))
    print()

    # ── Show last-known tunnel URL from .tunnel_url if it exists ─────────────
    for fname in (".tunnel_url", "tunnel_url.txt"):
        url_file = PROJECT_ROOT / fname
        if url_file.exists():
            try:
                saved_url = url_file.read_text(encoding="utf-8").splitlines()[0].strip()
                if saved_url.startswith("http"):
                    inner = saved_url.center(58)
                    box = (
                        "╔══════════════════════════════════════════════════════════╗\n"
                        "║         LAST SESSION — DASHBOARD URL (may still work)    ║\n"
                        "╠══════════════════════════════════════════════════════════╣\n"
                        f"║   {inner}   ║\n"
                        "╚══════════════════════════════════════════════════════════╝"
                    )
                    print(_cyan(box))
                    print(_bright_yellow(f"  ► {saved_url}"))
                    print(_dim("  (A fresh URL will be shown once the new tunnel starts)"))
                    print()
                    break
            except OSError:
                pass

    print(_dim("  Initializing logger and log directory..."))

    FIntelLogger.cleanup_old_logs(7)

    launcher_log = FIntelLogger.get_logger("Launcher")
    launcher_log.info("Launcher started")
    return launcher_log


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — EXPIRY INPUT
# ──────────────────────────────────────────────────────────────────────────────

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _prompt_date(label: str) -> str:
    """Loop until a valid YYYY-MM-DD date is entered."""
    while True:
        raw = input(f"  {label}: ").strip()
        if _DATE_RE.match(raw):
            try:
                datetime.date.fromisoformat(raw)
                return raw
            except ValueError:
                pass
        print(_red("  Invalid date. Use YYYY-MM-DD format."))


def _step2_expiry(launcher_log) -> tuple[str, str]:
    print()
    print(_white("─── Session Configuration ───"))
    print()

    near = _prompt_date("Near Expiry (weekly,  YYYY-MM-DD)")
    far  = _prompt_date("Far  Expiry (monthly, YYYY-MM-DD)")

    print()
    print(_green(f"  ✓ Expiries set — Near: {near}  Far: {far}"))

    session_data = {
        "near_expiry":   near,
        "far_expiry":    far,
        "session_start": datetime.datetime.now().isoformat(),
        "tunnel_url":    None,
    }
    session_path = PROJECT_ROOT / "fintel_session.json"
    session_path.write_text(json.dumps(session_data, indent=2), encoding="utf-8")

    launcher_log.info(f"Near expiry: {near}")
    launcher_log.info(f"Far expiry:  {far}")
    launcher_log.info("fintel_session.json written")

    return near, far


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — DATASERVER
# ──────────────────────────────────────────────────────────────────────────────

def _step3_dataserver(launcher_log) -> subprocess.Popen:
    print()
    print(_yellow("  [1/3] Starting DataHub (DataServer.py)..."))

    ds = _popen("DataServer.py")
    _start_stream_thread(ds, "DataServer")

    print(_green(f"  ✓ DataHub launched (PID: {ds.pid})"))
    launcher_log.info(f"DataServer launched PID={ds.pid}")

    print(_dim("  Waiting 4 s for Fyers WebSocket to initialize..."))
    time.sleep(4)

    return ds


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — CLOUDFLARE TUNNEL
# ──────────────────────────────────────────────────────────────────────────────

# Module-level mutable container for the discovered tunnel URL
_TUNNEL_URL_HOLDER: list[str | None] = [None]


def _step4_tunnel(launcher_log) -> subprocess.Popen:
    print()
    print(_yellow("  [3/3] Starting Cloudflare Tunnel (start_tunnel.py)..."))

    url_found_event = threading.Event()

    def url_callback(url: str) -> None:
        _TUNNEL_URL_HOLDER[0] = url
        url_found_event.set()

        # Persist tunnel_url.txt AND .tunnel_url (keep both in sync)
        ts_line = f"Session started: {datetime.datetime.now().isoformat()}"
        for fname in ("tunnel_url.txt", ".tunnel_url"):
            try:
                (PROJECT_ROOT / fname).write_text(
                    f"{url}\n{ts_line}", encoding="utf-8"
                )
            except OSError:
                pass

        # Update fintel_session.json
        try:
            session_path = PROJECT_ROOT / "fintel_session.json"
            if session_path.exists():
                data = json.loads(session_path.read_text(encoding="utf-8"))
                data["tunnel_url"] = url
                session_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            pass

        launcher_log.info(f"Tunnel URL detected: {url}")

    tunnel_proc = _popen("start_tunnel.py")
    _start_stream_thread(tunnel_proc, "Tunnel", url_callback=url_callback)

    print(_cyan(f"  ✓ Tunnel process launched (PID: {tunnel_proc.pid}). Waiting for URL..."))

    url_found_event.wait(timeout=30)

    url = _TUNNEL_URL_HOLDER[0]

    if url:
        inner = url.center(58)
        box = (
            "╔══════════════════════════════════════════════════════════╗\n"
            "║           F-INTEL DASHBOARD — LIVE ACCESS URL            ║\n"
            "╠══════════════════════════════════════════════════════════╣\n"
            f"║   {inner}   ║\n"
            "╠══════════════════════════════════════════════════════════╣\n"
            "║   Share this link to access the dashboard remotely       ║\n"
            "║   Link changes every session — copy it now               ║\n"
            "╚══════════════════════════════════════════════════════════╝"
        )
        print()
        print(_cyan(box))
        print()
        print(_bright_yellow(f"  ► COPY: {url}"))
        print(_dim("  (Also saved to tunnel_url.txt)"))
    else:
        # Build date/time strings for log path hint
        dt_str = datetime.datetime.now().strftime("%Y-%m-%d")
        ts_str = FIntelLogger._TIME_STR
        print()
        print(_red("  ⚠ WARNING: Tunnel URL not detected within 30s."))
        print(_red(f"    Check logs/{dt_str}/tunnel_{ts_str}.log for details."))
        launcher_log.warning("Tunnel URL not detected within 30s timeout")

    return tunnel_proc


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — VOLATILITY ANALYZER (interactive — no stdout pipe)
# ──────────────────────────────────────────────────────────────────────────────

def _step5_analyzer(launcher_log) -> subprocess.Popen:
    print()
    print(_yellow("  [2/3] Starting Volatility Analyzer (VolatilityAnalyzer.py)..."))
    print(_dim("  " + "─" * 55))
    print(_dim("  Note: VolatilityAnalyzer will ask for expiries — enter the same dates above."))
    print(_dim("  (Future update will read them automatically from fintel_session.json)"))
    print()

    va = subprocess.Popen(
        [str(PYTHON_EXE), "VolatilityAnalyzer.py"],
        cwd=str(PROJECT_ROOT),
    )
    launcher_log.info(f"VolatilityAnalyzer launched PID={va.pid}")
    return va


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — PROCESS MONITOR LOOP
# ──────────────────────────────────────────────────────────────────────────────

def _step6_monitor(ds_proc_ref: list, tunnel_proc: subprocess.Popen,
                   va_proc: subprocess.Popen, launcher_log) -> None:
    """Run in a daemon thread. Monitors process health every 5 s.

    ds_proc_ref is a mutable list [ds_process] so we can update the reference
    inside the thread when DataServer is restarted.
    """
    ds_restarted: bool = False

    while True:
        time.sleep(5)

        # ── VolatilityAnalyzer exit → clean shutdown ──────────────────────────
        if va_proc.poll() is not None:
            # Signal the main thread to proceed to shutdown
            # We raise a flag via a module-level event
            _shutdown_event.set()
            break

        # ── DataServer health ─────────────────────────────────────────────────
        ds = ds_proc_ref[0]
        if ds.poll() is not None:
            launcher_log.critical("DataServer exited unexpectedly. Attempting restart.")
            print(_red("  ✗ DataHub crashed! Restarting..."))

            if not ds_restarted:
                new_ds = _popen("DataServer.py")
                _start_stream_thread(new_ds, "DataServer")
                ds_proc_ref[0] = new_ds
                ds_restarted = True
                print(_green(f"  ✓ DataHub restarted (PID: {new_ds.pid})"))
                launcher_log.info(f"DataServer restarted PID={new_ds.pid}")
            else:
                launcher_log.critical("DataServer failed after restart. Manual intervention required.")
                print(_red("  ✗ DataHub failed again. Manual intervention required."))

        # ── Tunnel health ─────────────────────────────────────────────────────
        if tunnel_proc.poll() is not None:
            launcher_log.warning("Tunnel process exited.")
            print(_yellow("  ⚠ Tunnel process stopped. Remote access may be unavailable."))
            # Do not attempt automatic tunnel restart


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — GRACEFUL SHUTDOWN
# ──────────────────────────────────────────────────────────────────────────────

def _terminate_proc(proc: subprocess.Popen, label: str, launcher_log) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        launcher_log.warning(f"{label} did not terminate in 3 s — killing.")
        proc.kill()
    except OSError:
        pass


def _step7_shutdown(ds_proc_ref: list, tunnel_proc: subprocess.Popen,
                    launcher_log) -> None:
    print()
    print(_cyan("─── Shutting down F-Intel... ───"))

    _terminate_proc(ds_proc_ref[0], "DataServer", launcher_log)
    _terminate_proc(tunnel_proc,    "Tunnel",     launcher_log)

    summary = FIntelLogger.get_session_summary()
    lc = summary["lines_logged"]

    box = (
        "╔═══════════════════════════════╗\n"
        "║      SESSION SUMMARY          ║\n"
        "╠═══════════════════════════════╣\n"
        f"║  Started : {summary['session_start'][:19]}  ║\n"
        f"║  Log Dir : {str(summary['log_dir'])[-27:]:27s}  ║\n"
        f"║  Lines   : DataServer: {lc.get('DataServer', 0):<6d}  ║\n"
        f"║            Tunnel    : {lc.get('Tunnel', 0):<6d}  ║\n"
        f"║            Analyzer  : {lc.get('Analyzer', 0):<6d}  ║\n"
        f"║  Metrics : {summary['metrics_captured']:<6d} captured         ║\n"
        "╚═══════════════════════════════╝"
    )
    print()
    print(_cyan(box))
    print()

    launcher_log.info("Launcher shutdown complete.")


# ──────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SHUTDOWN EVENT (set by monitor thread when VA exits)
# ──────────────────────────────────────────────────────────────────────────────
_shutdown_event = threading.Event()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    session_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Step 1 — splash
    launcher_log = _step1_splash(session_ts)

    # Step 2 — expiry input
    _step2_expiry(launcher_log)

    # Step 3 — DataServer
    ds_proc = _step3_dataserver(launcher_log)
    ds_proc_ref: list = [ds_proc]  # mutable ref for monitor thread

    # Step 4 — VolatilityAnalyzer (interactive) — launched before tunnel
    va_proc = _step5_analyzer(launcher_log)

    # Step 5 — Tunnel (starts after Analyzer so the dashboard is up first)
    tunnel_proc = _step4_tunnel(launcher_log)

    # Step 6 — Monitor (daemon thread)
    monitor_thread = threading.Thread(
        target=_step6_monitor,
        args=(ds_proc_ref, tunnel_proc, va_proc, launcher_log),
        daemon=True,
    )
    monitor_thread.start()

    # Wait until VolatilityAnalyzer exits (signalled by monitor thread)
    _shutdown_event.wait()

    # Step 7 — Graceful shutdown
    _step7_shutdown(ds_proc_ref, tunnel_proc, launcher_log)

    sys.exit(0)


if __name__ == "__main__":
    main()
