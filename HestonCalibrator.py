"""
HestonCalibrator.py
===================
Background thread that re-calibrates Heston model parameters every 5 minutes
from the live option chain and writes results to SharedDataCache.

Market-hours only (09:15–15:30 IST). Uses only near-ATM CE strikes (6–12
strikes within 2% of spot) for speed — puts are handled via put-call parity
in the calibration objective.

Usage (in DataServer.py after hub.start()):
    from HestonCalibrator import HestonCalibrator
    calibrator = HestonCalibrator(shared_cache, hub.fyers)
    calibrator.start()
"""

import threading
import time
from datetime import datetime

try:
    import pytz
    _HAS_PYTZ = True
except ImportError:
    _HAS_PYTZ = False

from NiftyHestonMC import NiftyHestonMC, HestonMath

RISK_FREE = 0.07   # 7% Indian risk-free — must match StrategyEngine


class HestonCalibrator:
    """
    Background thread that recalibrates Heston params every 5 minutes
    from the live option chain and writes to SharedDataCache.
    Market hours only: 09:15 to 15:30 IST.
    """

    INTERVAL        = 300   # seconds between calibration runs
    STRIKES_NEAR_ATM = 12   # use up to 12 CE strikes within 2% of spot

    def __init__(self, shared_cache, fyers_instance):
        """
        Parameters
        ----------
        shared_cache    : SharedDataCache instance (from DataServer)
        fyers_instance  : authenticated fyersModel.FyersModel instance
        """
        self.cache = shared_cache
        self.fyers = fyers_instance
        self._stop = threading.Event()

        # Build a lightweight NiftyHestonMC for pure-math calls (no auth)
        self._mc = object.__new__(NiftyHestonMC)
        self._mc.fyers       = fyers_instance
        from OptionAnalytics import OptionAnalytics
        self._mc.analytics   = OptionAnalytics()
        self._mc.symbol      = "NSE:NIFTY50-INDEX"
        self._mc.spot_price  = 0
        self._mc.expiry_date = None
        self._mc.regimes     = {}

    # ── Market hours gate ─────────────────────────────────────────────────────

    def _in_market_hours(self) -> bool:
        """Return True if current IST time is between 09:15 and 15:30."""
        try:
            if _HAS_PYTZ:
                tz  = pytz.timezone('Asia/Kolkata')
                now = datetime.now(tz=tz)
            else:
                # Fallback: assume system is running in IST
                now = datetime.now()
            minutes = now.hour * 60 + now.minute
            return 9 * 60 + 15 <= minutes <= 15 * 60 + 30
        except Exception:
            return True   # fail open — attempt calibration

    # ── Single calibration run ────────────────────────────────────────────────

    def _calibrate_once(self):
        """
        Fetch spot + chain from SharedDataCache, select near-ATM CE strikes,
        run calibration, and write params back to cache.
        All exceptions are caught and logged — never propagate to caller.
        """
        try:
            # 1. Spot price
            spot = self.cache._spot
            if spot <= 0:
                print("[HestonCalibrator] Spot not available, skipping.")
                return

            # 2. Raw chain
            chain_data = self.cache.get_raw_chain()
            if not chain_data:
                print("[HestonCalibrator] Chain not available, skipping.")
                return

            # 3. Parse to DataFrame
            df = self._mc.parse_chain(chain_data)
            if df is None or df.empty:
                print("[HestonCalibrator] Parsed chain empty, skipping.")
                return

            # 4. Near-ATM CE subset for speed
            df = df.copy()
            df['dist'] = abs(df['strike'] - spot)
            subset = (
                df[(df['type'] == 'CE') & (df['dist'] < spot * 0.02)]
                .sort_values('dist')
                .head(self.STRIKES_NEAR_ATM)
            )
            if len(subset) < 3:
                print(f"[HestonCalibrator] Only {len(subset)} near-ATM CE "
                      "strikes — need ≥ 3, skipping.")
                return

            # 5. T (DTE in years)
            T = max(self.cache.get_T(), 0.001)

            # 6. Calibrate
            params = self._mc.calibrate_parameters(subset, spot, T, RISK_FREE)

            # 7. Store
            self.cache.set_heston_params(params)
            print(
                f"[HestonCalibrator] Calibrated OK — "
                f"kappa={params['kappa']:.2f} theta={params['theta']:.4f} "
                f"xi={params['xi']:.2f} rho={params['rho']:.2f} "
                f"v0={params['v0']:.4f}"
            )

        except Exception as e:
            print(f"[HestonCalibrator] Calibration failed (non-fatal): {e}")

    # ── Thread control ────────────────────────────────────────────────────────

    def start(self):
        """Start the background calibration loop."""
        def _loop():
            while not self._stop.is_set():
                if self._in_market_hours():
                    self._calibrate_once()
                else:
                    print("[HestonCalibrator] Outside market hours, sleeping.")
                self._stop.wait(self.INTERVAL)

        t = threading.Thread(target=_loop, name="HestonCalibrator", daemon=True)
        t.start()
        print("[HestonCalibrator] Background calibration thread started "
              f"(interval={self.INTERVAL}s).")

    def stop(self):
        """Signal the background thread to stop."""
        self._stop.set()
