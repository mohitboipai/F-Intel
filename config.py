"""
config.py — F-Intel Unified Configuration System
==================================================
Single source of truth for all market constants and calculation parameters.

Loads from fintel_config.json if present; uses scientifically-grounded defaults
otherwise.  Profile switching is in-memory only — saves to JSON on explicit
save_config() or via the /api/config endpoint in DataServer.

Scientific basis:
  risk_free_rate : RBI 91-day T-bill yield (Sep 2026 ≈ 5.26% p.a.)
                   → continuously compounded: ln(1.0526) ≈ 0.051274
                   Source: CCIL / RBI auctions (rbi.org.in)
                   Use T-bill, NOT Repo — T-bill reflects actual traded short-
                   term borrowing cost (Alice Blue, Quantsapp, CCIL convention).

  dividend_yield : NIFTY 50 trailing 12-month dividend yield (NSE Indices,
                   July 2026 = 1.22%).  Applied via Merton (1973) continuous
                   dividend BSM: d1 = [ln(S/K)+(r-q+0.5σ²)T] / (σ√T).
                   Gate: set q=0 when DTE ≤ dividend_dte_threshold (default 7)
                   because the dividend impact is negligible for short-dated
                   options and can introduce noise.

  nifty_lot_size : 65 (NSE revised August 2024 for NIFTY 50 F&O contracts).

  trading_days_year : 252 — used ONLY for HV/RV annualization (market
                       convention).  BSM time-to-expiry always uses calendar
                       days / days_in_year (365).  This deliberate split is
                       maintained per design (Q3).

Usage:
    import config
    r  = config.get("risk_free_rate")        # dynamic — respects active profile
    q  = config.get("dividend_yield")
    lot = config.get("nifty_lot_size")

    # OR — for one-liner backward-compat (values frozen at import time):
    from config import RISK_FREE_RATE, NIFTY_LOT_SIZE

    # Profile switch (in-memory, immediate):
    config.load_profile("0dte")

    # Save an override to disk:
    config.save_config({"risk_free_rate": 0.0513})
"""

from __future__ import annotations

import json
import math
import logging
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).parent / "fintel_config.json"
_log = logging.getLogger("fintel.config")

# ── Scientifically-grounded defaults ─────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    # ── Market ────────────────────────────────────────────────────────────────
    "nifty_lot_size"          : 65,       # NSE revised Aug 2024
    "nifty_strike_step"       : 50,       # NIFTY50 standard strike interval
    "symbol"                  : "NSE:NIFTY50-INDEX",

    # ── Pricing (BSM / Merton) ────────────────────────────────────────────────
    # RBI 91-day T-bill Sep 2026 ≈ 5.26% → ln(1.0526) = 0.051274 (cc)
    "risk_free_rate"          : round(math.log(1.0526), 6),   # 0.051274
    # NSE Indices NIFTY50 trailing dividend yield, Jul 2026
    "dividend_yield"          : 0.0122,
    "days_in_year"            : 365,      # calendar — for BSM T computation
    "trading_days_year"       : 252,      # trading — for HV/RV annualization
    # Gate: ignore dividend for very short DTE (effect < 0.02% of spot)
    "dividend_dte_threshold"  : 7,        # calendar days

    # ── GEX Calculation ───────────────────────────────────────────────────────
    "gex_move_pct"            : 0.01,     # 1% spot move convention
    "strike_filter_range"     : 0.05,     # ±5% of spot for chain filtering

    # ── IV Fallback (4-Tier) ──────────────────────────────────────────────────
    # Tier 4 flat fallback — used only when smile interpolation has < 3 points
    "iv_fallback_flat"        : 0.15,
    "iv_fallback_min"         : 0.08,     # floor: 8% IV (extreme compression)
    "iv_fallback_max"         : 0.80,     # ceiling: 80% IV (tail events)
    # EWMA ratio of IV/HV_20d — updated by regime engine; used in Tier 4
    "iv_hv_premium"           : 1.10,
    # Newton-Raphson starting guess for IV solver
    "iv_solver_seed"          : 0.15,

    # ── Vol Signal Parameters ─────────────────────────────────────────────────
    "ivp_lookback_days"       : 252,      # IV Percentile lookback (trading days)
    "ivr_lookback_days"       : 252,      # IV Rank lookback
    "vrp_zscore_window"       : 252,      # VRP z-score rolling window
    "hv_window_short"         : 5,        # RV short window
    "hv_window_medium"        : 20,       # RV primary window
    "hv_window_long"          : 60,       # RV long window
    # VER (Volatility Efficiency Ratio) thresholds — Park/C2C ratio
    "ver_high_threshold"      : 1.15,     # above → choppy / mean-reverting
    "ver_low_threshold"       : 0.80,     # below → trending / directional
    "vov_turbulent"           : 4.0,      # VoV (std of HV series) → turbulent
    # Consensus RV weights (Yang-Zhang 2000 → 40% min-variance; Garman-Klass,
    # Parkinson, C2C at 20% each)
    "rv_weight_yz"            : 0.40,
    "rv_weight_c2c"           : 0.20,
    "rv_weight_park"          : 0.20,
    "rv_weight_gk"            : 0.20,

    # ── Trade Execution Broadcasting ──────────────────────────────────────────
    "enable_trade_broadcast"  : False,    # Minion mode removed per user request
    "minion_endpoints"        : [],

    # ── Ignition Scanner (0DTE Gamma Compression → Ignition Detector) ─────────
    "ignition_scan_range_strikes"      : 6,       # ATM ± N strikes (both CE & PE)
    "ignition_compression_lookback"    : 8,       # bars for range_pct calculation
    "ignition_compression_percentile"  : 20,      # flag if below this %-ile of own dist
    "ignition_premium_ceiling"         : 40.0,    # max premium ₹ for compression filter
    "ignition_spot_atr_percentile"     : 25,      # underlying ATR gate percentile
    "ignition_spot_zscore_threshold"   : 2.0,     # spot ROC z-score for ignition trigger
    "ignition_volume_zscore_threshold" : 1.5,     # option volume z-score for confirmation
    "ignition_iv_spike_pct"            : 0.05,    # 5% IV acceleration threshold
    "ignition_entry_threshold"         : 65,      # confluence score to flag IGNITING
    "ignition_trail_stop_pct"          : 25,      # % giveback from HWM for runner trailing
    "ignition_max_candidates"          : 6,       # max ranked candidates returned
}

# ── Section mapping for structured JSON storage ───────────────────────────────
_SECTION_MAP: dict[str, str] = {
    "nifty_lot_size": "market", "nifty_strike_step": "market", "symbol": "market",
    "risk_free_rate": "pricing", "dividend_yield": "pricing",
    "days_in_year": "pricing", "trading_days_year": "pricing",
    "dividend_dte_threshold": "pricing",
    "gex_move_pct": "gex", "strike_filter_range": "gex",
    "iv_fallback_flat": "iv_fallback", "iv_fallback_min": "iv_fallback",
    "iv_fallback_max": "iv_fallback", "iv_hv_premium": "iv_fallback",
    "iv_solver_seed": "iv_fallback",
    "ivp_lookback_days": "vol_signals", "ivr_lookback_days": "vol_signals",
    "vrp_zscore_window": "vol_signals",
    "hv_window_short": "vol_signals", "hv_window_medium": "vol_signals",
    "hv_window_long": "vol_signals",
    "ver_high_threshold": "vol_signals", "ver_low_threshold": "vol_signals",
    "vov_turbulent": "vol_signals",
    "rv_weight_yz": "vol_signals", "rv_weight_c2c": "vol_signals",
    "rv_weight_park": "vol_signals", "rv_weight_gk": "vol_signals",
    "enable_trade_broadcast": "execution", "minion_endpoints": "execution",
    "ignition_scan_range_strikes": "ignition", "ignition_compression_lookback": "ignition",
    "ignition_compression_percentile": "ignition", "ignition_premium_ceiling": "ignition",
    "ignition_spot_atr_percentile": "ignition", "ignition_spot_zscore_threshold": "ignition",
    "ignition_volume_zscore_threshold": "ignition", "ignition_iv_spike_pct": "ignition",
    "ignition_entry_threshold": "ignition", "ignition_trail_stop_pct": "ignition",
    "ignition_max_candidates": "ignition",
}

# ── Active config state (in-memory; may differ from disk after load_profile) ──
_active: dict[str, Any] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Internal loader
# ─────────────────────────────────────────────────────────────────────────────

def _load() -> dict[str, Any]:
    """Load config from JSON file (if present), merging over defaults."""
    cfg: dict[str, Any] = dict(_DEFAULTS)
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
                saved: dict[str, Any] = json.load(fh)
            # Merge all sections flat
            for section in ("market", "pricing", "gex", "iv_fallback", "vol_signals"):
                cfg.update(saved.get(section, {}))
            cfg["_active_profile"] = "default"
            _log.info("fintel_config.json loaded (profile: default)")
        except Exception as exc:
            _log.warning(f"Config load error ({exc}); using built-in defaults")
    else:
        cfg["_active_profile"] = "default"
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def reload() -> None:
    """Re-read fintel_config.json from disk into the active in-memory state."""
    global _active
    _active = _load()


def get(key: str, default: Any = None) -> Any:
    """Return the active value for *key*, falling back to the built-in default."""
    return _active.get(key, _DEFAULTS.get(key, default))


def active_profile() -> str:
    """Return the name of the currently active profile."""
    return str(_active.get("_active_profile", "default"))


def list_profiles() -> list[str]:
    """Return names of all profiles defined in fintel_config.json."""
    if not _CONFIG_PATH.exists():
        return ["default"]
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        return ["default"] + list(saved.get("profiles", {}).keys())
    except Exception:
        return ["default"]


def load_profile(name: str) -> bool:
    """
    Switch to a named profile (in-memory only — does NOT write to disk).

    The profile values are overlaid on top of the base config loaded from
    fintel_config.json sections, so a profile only needs to specify overrides.
    """
    global _active
    if name == "default":
        _active = _load()
        _active["_active_profile"] = "default"
        _log.info("Switched to profile: default")
        return True

    if not _CONFIG_PATH.exists():
        _log.warning("No fintel_config.json found; cannot switch profile")
        return False

    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
    except Exception as exc:
        _log.error(f"Cannot read config file: {exc}")
        return False

    profiles: dict[str, Any] = saved.get("profiles", {})
    if name not in profiles:
        available = list(profiles.keys())
        _log.error(f"Profile '{name}' not found. Available: {available}")
        return False

    # Reload base first, then overlay profile
    base = _load()
    base.update(profiles[name])
    base["_active_profile"] = name
    _active = base
    _log.info(f"Switched to profile: {name}")
    return True


def save_config(overrides: dict[str, Any], profile: str | None = None) -> None:
    """
    Persist *overrides* to fintel_config.json (non-destructive merge).

    If *profile* is given, overrides are saved under that profile block instead
    of the base sections.  Also updates the in-memory _active state.
    """
    global _active
    saved: dict[str, Any] = {}
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except Exception:
            pass

    if profile:
        saved.setdefault("profiles", {}).setdefault(profile, {}).update(overrides)
    else:
        for key, val in overrides.items():
            section = _SECTION_MAP.get(key, "vol_signals")
            saved.setdefault(section, {})[key] = val

    # Update in-memory state as well
    _active.update(overrides)

    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(saved, fh, indent=2)
        _log.info(f"Config saved: {list(overrides.keys())}")
    except Exception as exc:
        _log.error(f"Cannot write config file: {exc}")


def as_dict() -> dict[str, Any]:
    """Return a snapshot of the current active config (excludes internal keys)."""
    return {k: v for k, v in _active.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants (frozen at import time for backward-compat imports)
# For runtime-updated values after load_profile(), always use config.get()
# ─────────────────────────────────────────────────────────────────────────────

# Bootstrap on first import
reload()

NIFTY_LOT_SIZE         = _active.get("nifty_lot_size",         65)
NIFTY_STRIKE_STEP      = _active.get("nifty_strike_step",       50)
RISK_FREE_RATE         = _active.get("risk_free_rate",          round(math.log(1.0526), 6))
DIVIDEND_YIELD         = _active.get("dividend_yield",          0.0122)
DAYS_IN_YEAR           = _active.get("days_in_year",            365)
TRADING_DAYS_YEAR      = _active.get("trading_days_year",       252)
DIVIDEND_DTE_THRESHOLD = _active.get("dividend_dte_threshold",  7)
GEX_MOVE_PCT           = _active.get("gex_move_pct",            0.01)
STRIKE_FILTER_RANGE    = _active.get("strike_filter_range",     0.05)
IV_FALLBACK_FLAT       = _active.get("iv_fallback_flat",        0.15)
IV_FALLBACK_MIN        = _active.get("iv_fallback_min",         0.08)
IV_FALLBACK_MAX        = _active.get("iv_fallback_max",         0.80)
IV_HV_PREMIUM          = _active.get("iv_hv_premium",           1.10)
IV_SOLVER_SEED         = _active.get("iv_solver_seed",          0.15)
IVP_LOOKBACK_DAYS      = _active.get("ivp_lookback_days",       252)
VER_HIGH_THRESHOLD     = _active.get("ver_high_threshold",      1.15)
VER_LOW_THRESHOLD      = _active.get("ver_low_threshold",       0.80)
VOV_TURBULENT          = _active.get("vov_turbulent",           4.0)
