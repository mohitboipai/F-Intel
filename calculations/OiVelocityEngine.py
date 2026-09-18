"""
OiVelocityEngine.py — Multi-Timeframe Open Interest Velocity & Intraday Flow Engine.

Computes real-time OI change rates (1m, 3m, 5m, 15m), absolute contracts/lots shifted
(Baseline OI -> Current OI, Net ΔOI, % Change), historical rewind comparison,
major level status (Call Wall, Put Wall, ATM), outlier surge hotspots, and
focused institutional flow advisory.
"""

import time
import datetime
from typing import Dict, List, Any, Optional, Tuple


class OiVelocityEngine:
    TIMEFRAMES = {
        '1m': 60,
        '3m': 180,
        '5m': 300,
        '15m': 900
    }

    def __init__(self, lot_size: int = 65):
        self.lot_size = lot_size

    def calculate_velocity(
        self,
        snaps: List[Dict[str, Any]],
        timeframe: str = '5m',
        rewind_ts: Optional[float] = None,
        spot: float = 0.0,
        strike_range_pts: Optional[int] = 1000
    ) -> Dict[str, Any]:
        """
        Compute multi-timeframe per-strike OI velocity, absolute changes, and major level hotspots.

        snaps: list of {'ts': float, 'time_str': str, 'spot': float, 'oi_map': {(strike, type): int}}
        timeframe: '1m', '3m', '5m', or '15m'
        rewind_ts: epoch timestamp to rewind history to. If None, uses latest snapshot.
        spot: current underlying spot price.
        strike_range_pts: points around spot to display (default 1000 pts = ±20 strikes). None for all.
        """
        if not snaps or len(snaps) < 2:
            return {
                'ok': False,
                'status': 'INSUFFICIENT_DATA',
                'message': 'Accumulating OI snapshots (requires at least 2 ticks)...',
                'strikes': [],
                'history_index': [],
                'available_history_min': 0,
                'analysis': self._get_empty_analysis(spot)
            }

        window_secs = self.TIMEFRAMES.get(timeframe, 300)
        latest_snap = snaps[-1]
        live_ts = latest_snap['ts']
        live_time_str = latest_snap.get('time_str', datetime.datetime.fromtimestamp(live_ts).strftime('%H:%M:%S'))

        available_history_sec = max(live_ts - snaps[0]['ts'], 0)
        available_history_min = round(available_history_sec / 60.0, 1)

        # Build history index for time scrubber
        history_index = [
            {
                'ts': s['ts'],
                'time_str': s.get('time_str', datetime.datetime.fromtimestamp(s['ts']).strftime('%H:%M:%S')),
                'spot': s.get('spot', spot)
            }
            for s in snaps
        ]

        # Determine active snapshot (rewind vs live)
        is_rewound = False
        active_snap = latest_snap
        if rewind_ts is not None and len(snaps) > 1:
            try:
                rewind_ts_f = float(rewind_ts)
                closest = min(snaps, key=lambda s: abs(s['ts'] - rewind_ts_f))
                if abs(closest['ts'] - live_ts) > 12:
                    active_snap = closest
                    is_rewound = True
            except Exception:
                pass

        active_ts = active_snap['ts']
        active_time_str = active_snap.get('time_str', datetime.datetime.fromtimestamp(active_ts).strftime('%H:%M:%S'))
        active_spot = active_snap.get('spot', spot) or spot

        # Compute velocity and absolute delta for active snapshot
        active_res = self._compute_single_snapshot_velocity(
            snaps, active_snap, window_secs, active_spot, strike_range_pts
        )

        # If rewound, also compute live snapshot velocity for ghost comparison on same graph
        live_res = None
        if is_rewound:
            live_res = self._compute_single_snapshot_velocity(
                snaps, latest_snap, window_secs, spot, strike_range_pts
            )

        # Multi-timeframe trend matrix (1m, 3m, 5m, 15m)
        tf_matrix = {}
        for tf_key, tf_secs in self.TIMEFRAMES.items():
            tf_data = self._compute_single_snapshot_velocity(
                snaps, active_snap, tf_secs, active_spot, strike_range_pts
            )
            tf_matrix[tf_key] = {
                'window_secs': tf_secs,
                'elapsed_min': tf_data['elapsed_min'],
                'net_vel': tf_data['total_net_vel'],
                'ce_vel': tf_data['total_ce_vel'],
                'pe_vel': tf_data['total_pe_vel'],
                'net_delta': tf_data['net_delta_contracts'],
                'ce_delta': tf_data['total_ce_delta'],
                'pe_delta': tf_data['total_pe_delta'],
                'bias': 'BULLISH' if tf_data['total_net_vel'] > 2000 else ('BEARISH' if tf_data['total_net_vel'] < -2000 else 'NEUTRAL'),
                'bias_color': '#00e676' if tf_data['total_net_vel'] > 2000 else ('#ff3366' if tf_data['total_net_vel'] < -2000 else '#94a3b8'),
            }

        # Multi-timeframe & major levels advisory synthesis
        analysis = self._generate_advisory_analysis(active_res, tf_matrix, active_spot, timeframe, is_rewound)

        return {
            'ok': True,
            'timeframe': timeframe,
            'window_secs': window_secs,
            'elapsed_min': active_res['elapsed_min'],
            'spot': active_spot,
            'is_rewound': is_rewound,
            'rewind_ts': active_ts,
            'rewind_time_str': active_time_str,
            'live_ts': live_ts,
            'live_time_str': live_time_str,
            'available_history_min': available_history_min,
            'history_index': history_index,
            'strikes': active_res['strikes'],
            'comparison_live_strikes': live_res['strikes'] if live_res else None,
            'major_levels': analysis.get('major_levels', {}),
            'hotspots': analysis.get('hotspots', []),
            'analysis': analysis,
            'total_call_oi': active_res.get('total_call_oi', 0),
            'total_put_oi': active_res.get('total_put_oi', 0),
            'day_pcr': active_res.get('day_pcr', 1.0),
            'chain_total_ce_oi': active_res.get('chain_total_ce_oi', 0),
            'chain_total_pe_oi': active_res.get('chain_total_pe_oi', 0),
            'chain_pcr': active_res.get('chain_pcr', 1.0),
            'call_flow': analysis.get('call_flow', {}),
            'put_flow': analysis.get('put_flow', {})
        }

    def _compute_single_snapshot_velocity(
        self,
        snaps: List[Dict[str, Any]],
        active_snap: Dict[str, Any],
        window_secs: int,
        spot: float,
        strike_range_pts: Optional[int] = 1000
    ) -> Dict[str, Any]:
        """Compute per-strike baseline, current, delta, lot count, and velocity."""
        now = active_snap['ts']
        target_time = now - window_secs

        candidates = [s for s in snaps if s['ts'] < now]
        if not candidates:
            candidates = [s for s in snaps if s != active_snap]
        if not candidates:
            baseline = active_snap
        else:
            baseline = min(candidates, key=lambda s: abs(s['ts'] - target_time))

        elapsed_min = max((now - baseline['ts']) / 60.0, 0.05)

        active_oi = active_snap.get('oi_map')
        if not active_oi and 'strikes' in active_snap:
            active_oi = {}
            for st in active_snap['strikes']:
                s = float(st.get('strike', 0))
                if 'call_oi' in st:
                    active_oi[(s, 'CE')] = int(st.get('call_oi', 0) or 0)
                if 'put_oi' in st:
                    active_oi[(s, 'PE')] = int(st.get('put_oi', 0) or 0)
        elif not active_oi:
            active_oi = {}

        base_oi = baseline.get('oi_map')
        if not base_oi and 'strikes' in baseline:
            base_oi = {}
            for st in baseline['strikes']:
                s = float(st.get('strike', 0))
                if 'call_oi' in st:
                    base_oi[(s, 'CE')] = int(st.get('call_oi', 0) or 0)
                if 'put_oi' in st:
                    base_oi[(s, 'PE')] = int(st.get('put_oi', 0) or 0)
        elif not base_oi:
            base_oi = {}

        # Extract all unique strikes
        all_strikes = set()
        for k in active_oi.keys():
            if isinstance(k, (tuple, list)):
                all_strikes.add(float(k[0]))
            elif isinstance(k, (int, float)):
                all_strikes.add(float(k))
        for k in base_oi.keys():
            if isinstance(k, (tuple, list)):
                all_strikes.add(float(k[0]))
            elif isinstance(k, (int, float)):
                all_strikes.add(float(k))

        sorted_strikes = sorted(list(all_strikes))

        # Filter strikes if strike_range_pts is specified
        if strike_range_pts and strike_range_pts > 0 and spot > 0:
            band_lo = spot - strike_range_pts
            band_hi = spot + strike_range_pts
            atm_strikes = [s for s in sorted_strikes if band_lo <= s <= band_hi]
            if len(atm_strikes) >= 6:
                sorted_strikes = atm_strikes

        strikes_data = []
        total_ce_vel = 0
        total_pe_vel = 0
        total_net_vel = 0
        total_ce_delta = 0
        total_pe_delta = 0

        for st in sorted_strikes:
            ce_key = (st, 'CE')
            pe_key = (st, 'PE')

            ce_now = active_oi.get(ce_key, 0)
            pe_now = active_oi.get(pe_key, 0)

            ce_base = base_oi.get(ce_key, ce_now)
            pe_base = base_oi.get(pe_key, pe_now)

            ce_delta = ce_now - ce_base
            pe_delta = pe_now - pe_base

            ce_pct = round((ce_delta / ce_base * 100), 1) if ce_base > 0 else 0.0
            pe_pct = round((pe_delta / pe_base * 100), 1) if pe_base > 0 else 0.0

            ce_lots = int(round(ce_delta / self.lot_size))
            pe_lots = int(round(pe_delta / self.lot_size))

            ce_vel = int(round(ce_delta / elapsed_min))
            pe_vel = int(round(pe_delta / elapsed_min))
            net_vel = pe_vel - ce_vel  # PE building (+) or CE unwinding (+) = bullish
            net_delta = pe_delta - ce_delta

            total_ce_vel += ce_vel
            total_pe_vel += pe_vel
            total_net_vel += net_vel
            total_ce_delta += ce_delta
            total_pe_delta += pe_delta

            # Flow Signal Classification
            signal = "BALANCED"
            if ce_delta > 20000 and pe_delta > 20000:
                signal = "STRADDLE_BUILDING"
            elif ce_delta > 25000:
                signal = "CALL_WRITING_SURGE"
            elif pe_delta > 25000:
                signal = "PUT_WRITING_SURGE"
            elif ce_delta < -15000:
                signal = "CALL_SHORT_COVERING"
            elif pe_delta < -15000:
                signal = "PUT_CAPITULATION"

            strikes_data.append({
                'strike': st,
                # Absolute Values (How much it changed)
                'call_base_oi': ce_base,
                'call_curr_oi': ce_now,
                'call_delta': ce_delta,
                'call_pct': ce_pct,
                'call_lots': ce_lots,
                'call_oi': ce_now,
                'put_base_oi': pe_base,
                'put_curr_oi': pe_now,
                'put_delta': pe_delta,
                'put_pct': pe_pct,
                'put_lots': pe_lots,
                'put_oi': pe_now,
                'net_delta': net_delta,
                # Velocity Rates (Per min)
                'call_vel': ce_vel,
                'put_vel': pe_vel,
                'net_vel': net_vel,
                'signal': signal,
                # Flags for coloring
                'call_bar_len': abs(ce_delta),
                'call_vel_bar_len': abs(ce_vel),
                'call_is_unwinding': ce_delta < 0,
                'put_bar_len': abs(pe_delta),
                'put_vel_bar_len': abs(pe_vel),
                'put_is_unwinding': pe_delta < 0
            })

        # Full-chain total OI across all available strikes in the active snapshot
        chain_total_ce_oi = sum(int(v) for (stk, opt_type), v in active_oi.items() if opt_type == 'CE')
        chain_total_pe_oi = sum(int(v) for (stk, opt_type), v in active_oi.items() if opt_type == 'PE')
        chain_pcr = round(chain_total_pe_oi / chain_total_ce_oi, 2) if chain_total_ce_oi > 0 else 1.0

        # Visible strikes total OI (within selected range)
        visible_total_ce_oi = sum(s['call_curr_oi'] for s in strikes_data)
        visible_total_pe_oi = sum(s['put_curr_oi'] for s in strikes_data)
        visible_pcr = round(visible_total_pe_oi / visible_total_ce_oi, 2) if visible_total_ce_oi > 0 else 1.0

        tot_ce = chain_total_ce_oi if chain_total_ce_oi > 0 else visible_total_ce_oi
        tot_pe = chain_total_pe_oi if chain_total_pe_oi > 0 else visible_total_pe_oi
        pcr = chain_pcr if chain_total_ce_oi > 0 else visible_pcr

        return {
            'elapsed_min': round(elapsed_min, 2),
            'strikes': strikes_data,
            'total_ce_vel': total_ce_vel,
            'total_pe_vel': total_pe_vel,
            'total_net_vel': total_net_vel,
            'total_ce_delta': total_ce_delta,
            'total_pe_delta': total_pe_delta,
            'net_delta_contracts': total_pe_delta - total_ce_delta,
            'chain_total_ce_oi': chain_total_ce_oi,
            'chain_total_pe_oi': chain_total_pe_oi,
            'chain_pcr': chain_pcr,
            'visible_total_ce_oi': visible_total_ce_oi,
            'visible_total_pe_oi': visible_total_pe_oi,
            'visible_pcr': visible_pcr,
            'total_call_oi': tot_ce,
            'total_put_oi': tot_pe,
            'day_pcr': pcr
        }

    def _generate_advisory_analysis(
        self,
        active_res: Dict[str, Any],
        tf_matrix: Dict[str, Any],
        spot: float,
        timeframe: str,
        is_rewound: bool
    ) -> Dict[str, Any]:
        """
        Synthesize institutional quantitative flow advisory focused strictly on:
        1. Major Levels (Call Wall, Put Wall, ATM Strike)
        2. Outlier Hotspots (Where major changes happened)
        """
        strikes = active_res.get('strikes', [])
        if not strikes:
            return self._get_empty_analysis(spot)

        # ── 1. MAJOR LEVELS IDENTIFICATION ─────────────────────────────────
        # Call Wall = Peak Call Open Interest
        call_wall_item = max(strikes, key=lambda s: s['call_oi'], default=None)
        # Put Wall = Peak Put Open Interest
        put_wall_item = max(strikes, key=lambda s: s['put_oi'], default=None)
        # ATM Strike = Strike closest to spot
        atm_item = min(strikes, key=lambda s: abs(s['strike'] - spot), default=None) if spot > 0 else None

        major_levels = {}
        if call_wall_item:
            cw_delta = call_wall_item['call_delta']
            cw_status = "STRENGTHENING (Adding Resistance)" if cw_delta > 10000 else (
                "WEAKENING (Short Covering)" if cw_delta < -10000 else "STEADY"
            )
            major_levels['call_wall'] = {
                'strike': call_wall_item['strike'],
                'total_oi': call_wall_item['call_oi'],
                'base_oi': call_wall_item['call_base_oi'],
                'delta': cw_delta,
                'pct': call_wall_item['call_pct'],
                'lots': call_wall_item['call_lots'],
                'status': cw_status,
                'status_color': '#ff3366' if cw_delta > 10000 else ('#00e5ff' if cw_delta < -10000 else '#94a3b8')
            }

        if put_wall_item:
            pw_delta = put_wall_item['put_delta']
            pw_status = "STRENGTHENING (Building Floor)" if pw_delta > 10000 else (
                "WEAKENING (Long Capitulation)" if pw_delta < -10000 else "STEADY"
            )
            major_levels['put_wall'] = {
                'strike': put_wall_item['strike'],
                'total_oi': put_wall_item['put_oi'],
                'base_oi': put_wall_item['put_base_oi'],
                'delta': pw_delta,
                'pct': put_wall_item['put_pct'],
                'lots': put_wall_item['put_lots'],
                'status': pw_status,
                'status_color': '#00e676' if pw_delta > 10000 else ('#ff9100' if pw_delta < -10000 else '#94a3b8')
            }

        if atm_item:
            atm_straddle_delta = atm_item['call_delta'] + atm_item['put_delta']
            atm_status = "VOL ABSORPTION (Straddle Writing)" if atm_straddle_delta > 15000 else (
                "RANGE EXPANSION (Straddle Unwinding)" if atm_straddle_delta < -15000 else "BALANCED"
            )
            major_levels['atm'] = {
                'strike': atm_item['strike'],
                'spot': spot,
                'call_delta': atm_item['call_delta'],
                'put_delta': atm_item['put_delta'],
                'straddle_delta': atm_straddle_delta,
                'status': atm_status,
                'status_color': '#ffd54f' if abs(atm_straddle_delta) > 15000 else '#94a3b8'
            }

        # ── 2. OUTLIER HOTSPOTS & FLOW CONCENTRATION ──────────────────────
        hotspots = []

        max_ce_write = max(strikes, key=lambda s: s['call_delta'], default=None)
        max_pe_write = max(strikes, key=lambda s: s['put_delta'], default=None)
        max_ce_unwind = min(strikes, key=lambda s: s['call_delta'], default=None)
        max_pe_unwind = min(strikes, key=lambda s: s['put_delta'], default=None)

        # Check for Dual Writing at same strike (Straddle Pinning / Premium Harvest)
        is_dual_write = bool(
            max_ce_write and max_pe_write
            and max_ce_write['call_delta'] > 5000
            and max_pe_write['put_delta'] > 5000
            and abs(max_ce_write['strike'] - max_pe_write['strike']) <= 50
        )

        # Check for Dual Unwinding at same strike (Straddle Closure / Volatility Exit)
        is_dual_unwind = bool(
            max_ce_unwind and max_pe_unwind
            and max_ce_unwind['call_delta'] < -5000
            and max_pe_unwind['put_delta'] < -5000
            and abs(max_ce_unwind['strike'] - max_pe_unwind['strike']) <= 50
        )

        # A. Dual Writing or Individual Writing Surges
        if is_dual_write and max_ce_write and max_pe_write:
            hotspots.append({
                'type': 'STRADDLE_PINNING',
                'strike': max_ce_write['strike'],
                'option_type': 'STRADDLE',
                'call_delta': max_ce_write['call_delta'],
                'put_delta': max_pe_write['put_delta'],
                'delta': max_ce_write['call_delta'] + max_pe_write['put_delta'],
                'lots': max_ce_write['call_lots'] + max_pe_write['put_lots'],
                'title': f"STRADDLE PIN ACCUMULATION ({max_ce_write['strike']:.0f})",
                'desc': f"Dual-wing writing at {max_ce_write['strike']:.0f}: +{max_ce_write['call_delta']:,} CE & +{max_pe_write['put_delta']:,} PE added. Range compression / pinning.",
                'color': '#ffd54f'
            })
        else:
            if max_ce_write and max_ce_write['call_delta'] > 5000:
                hotspots.append({
                    'type': 'CALL_WRITING_SURGE',
                    'strike': max_ce_write['strike'],
                    'option_type': 'CE',
                    'delta': max_ce_write['call_delta'],
                    'lots': max_ce_write['call_lots'],
                    'base_oi': max_ce_write['call_base_oi'],
                    'curr_oi': max_ce_write['call_curr_oi'],
                    'pct': max_ce_write['call_pct'],
                    'vel': max_ce_write['call_vel'],
                    'title': f"CALL CEILING DEFENSE (+{max_ce_write['call_delta']:,})",
                    'desc': f"Dealers added +{max_ce_write['call_delta']:,} contracts ({max_ce_write['call_pct']:+.1f}%) at {max_ce_write['strike']:.0f} CE",
                    'color': '#ff3366'
                })

            if max_pe_write and max_pe_write['put_delta'] > 5000:
                hotspots.append({
                    'type': 'PUT_WRITING_SURGE',
                    'strike': max_pe_write['strike'],
                    'option_type': 'PE',
                    'delta': max_pe_write['put_delta'],
                    'lots': max_pe_write['put_lots'],
                    'base_oi': max_pe_write['put_base_oi'],
                    'curr_oi': max_pe_write['put_curr_oi'],
                    'pct': max_pe_write['put_pct'],
                    'vel': max_pe_write['put_vel'],
                    'title': f"PUT FLOOR DEFENSE (+{max_pe_write['put_delta']:,})",
                    'desc': f"Bulls added +{max_pe_write['put_delta']:,} contracts ({max_pe_write['put_pct']:+.1f}%) at {max_pe_write['strike']:.0f} PE",
                    'color': '#00e676'
                })

        # B. Dual Unwinding (Straddle Exit) or Individual Unwinding Surges
        if is_dual_unwind and max_ce_unwind and max_pe_unwind:
            hotspots.append({
                'type': 'STRADDLE_UNWINDING',
                'strike': max_ce_unwind['strike'],
                'option_type': 'STRADDLE',
                'call_delta': max_ce_unwind['call_delta'],
                'put_delta': max_pe_unwind['put_delta'],
                'delta': max_ce_unwind['call_delta'] + max_pe_unwind['put_delta'],
                'lots': max_ce_unwind['call_lots'] + max_pe_unwind['put_lots'],
                'title': f"STRADDLE POSITION CLOSURE ({max_ce_unwind['strike']:.0f})",
                'desc': f"Simultaneous Call ({max_ce_unwind['call_delta']:,}) & Put ({max_pe_unwind['put_delta']:,}) liquidation at {max_ce_unwind['strike']:.0f}: Institutional straddle unwind / profit booking (Neutral flow reduction, not directional).",
                'color': '#c084fc'
            })
        else:
            if max_ce_unwind and max_ce_unwind['call_delta'] < -5000:
                hotspots.append({
                    'type': 'CALL_SHORT_COVERING',
                    'strike': max_ce_unwind['strike'],
                    'option_type': 'CE',
                    'delta': max_ce_unwind['call_delta'],
                    'lots': max_ce_unwind['call_lots'],
                    'base_oi': max_ce_unwind['call_base_oi'],
                    'curr_oi': max_ce_unwind['call_curr_oi'],
                    'pct': max_ce_unwind['call_pct'],
                    'vel': max_ce_unwind['call_vel'],
                    'title': f"SHORT SQUEEZE SURGE ({max_ce_unwind['call_delta']:,})",
                    'desc': f"Call writers panicked & covered {max_ce_unwind['call_delta']:,} contracts at {max_ce_unwind['strike']:.0f} CE",
                    'color': '#00e5ff'
                })

            if max_pe_unwind and max_pe_unwind['put_delta'] < -5000:
                hotspots.append({
                    'type': 'PUT_CAPITULATION',
                    'strike': max_pe_unwind['strike'],
                    'option_type': 'PE',
                    'delta': max_pe_unwind['put_delta'],
                    'lots': max_pe_unwind['put_lots'],
                    'base_oi': max_pe_unwind['put_base_oi'],
                    'curr_oi': max_pe_unwind['put_curr_oi'],
                    'pct': max_pe_unwind['put_pct'],
                    'vel': max_pe_unwind['put_vel'],
                    'title': f"PUT LIQUIDATION ({max_pe_unwind['put_delta']:,})",
                    'desc': f"Support removed: put writers shed {max_pe_unwind['put_delta']:,} contracts at {max_pe_unwind['strike']:.0f} PE",
                    'color': '#ff9100'
                })

        net_vel = active_res['total_net_vel']
        ce_vel = active_res['total_ce_vel']
        pe_vel = active_res['total_pe_vel']
        net_delta = active_res['net_delta_contracts']
        ce_delta = active_res['total_ce_delta']
        pe_delta = active_res['total_pe_delta']

        # Determine Overall Flow Regime
        if is_dual_unwind and abs(net_delta) < 15000:
            regime = "STRADDLE VOLATILITY UNWINDING (NEUTRAL DECOMPRESSION)"
            bias = "NEUTRAL (CLOSURE)"
            bias_color = "#c084fc"
        elif is_dual_write and abs(net_delta) < 15000:
            regime = "DUAL-WING STRADDLE PINNING (PREMIUM HARVEST)"
            bias = "NEUTRAL PIN"
            bias_color = "#ffd54f"
        elif pe_delta > ce_delta * 1.4 and pe_delta > 15000:
            regime = "STRONG PUT WRITING ACCUMULATION (BULLISH)"
            bias = "BULLISH"
            bias_color = "#00e676"
        elif ce_delta > pe_delta * 1.4 and ce_delta > 15000:
            regime = "HEAVY CALL OVERHEAD WRITING (BEARISH)"
            bias = "BEARISH"
            bias_color = "#ff3366"
        elif ce_delta < -10000 and pe_delta >= 0:
            regime = "SHORT SQUEEZE MOMENTUM (CALL COVERING)"
            bias = "BULLISH SQUEEZE"
            bias_color = "#00e5ff"
        elif pe_delta < -10000 and ce_delta >= 0:
            regime = "LONG LIQUIDATION CASCADE (PUT DUMP)"
            bias = "BEARISH CASCADE"
            bias_color = "#ff9100"
        elif ce_delta > 12000 and pe_delta > 12000:
            regime = "DUAL-WING PREMIUM HARVESTING (PINNING)"
            bias = "NEUTRAL / PIN"
            bias_color = "#ffd54f"
        else:
            regime = "BALANCED ROTATIONAL FLOW"
            bias = "NEUTRAL"
            bias_color = "#94a3b8"

        # Construct crisp executive narrative focused on major shifts without conflicting directional alerts
        narrative_parts = []
        if is_rewound:
            narrative_parts.append("[HISTORICAL REWIND]")

        if is_dual_unwind and max_ce_unwind and max_pe_unwind:
            narrative_parts.append(
                f"⚡ Straddle Unwinding at {max_ce_unwind['strike']:.0f}: Both CE ({max_ce_unwind['call_delta']:,}) & PE ({max_pe_unwind['put_delta']:,}) closed simultaneously — institutional position exit/roll, not a directional breakout."
            )
        elif is_dual_write and max_ce_write and max_pe_write:
            narrative_parts.append(
                f"⚡ Straddle Pinning at {max_ce_write['strike']:.0f}: Heavy dual writing (CE +{max_ce_write['call_delta']:,}, PE +{max_pe_write['put_delta']:,}) locking spot into a narrow pinning range."
            )
        else:
            if major_levels.get('put_wall') and major_levels['put_wall']['delta'] > 10000:
                pw = major_levels['put_wall']
                narrative_parts.append(f"Put Wall at {pw['strike']:.0f} strengthened by +{pw['delta']:,} contracts.")
            elif major_levels.get('call_wall') and major_levels['call_wall']['delta'] > 10000:
                cw = major_levels['call_wall']
                narrative_parts.append(f"Call Wall at {cw['strike']:.0f} reinforced by +{cw['delta']:,} contracts.")

            if hotspots:
                top_spot = hotspots[0]
                narrative_parts.append(f"Major shift: {top_spot['desc']}.")
            else:
                narrative_parts.append(f"Net volume shift: {net_delta:+,} contracts across active strikes.")

        narrative = " ".join(narrative_parts)

        # Aliases and unified dictionary contract
        top_ce_write_stk = max_ce_write['strike'] if (max_ce_write and max_ce_write['call_delta'] > 5000) else None
        top_pe_write_stk = max_pe_write['strike'] if (max_pe_write and max_pe_write['put_delta'] > 5000) else None
        top_ce_unwind_stk = max_ce_unwind['strike'] if (max_ce_unwind and max_ce_unwind['call_delta'] < -5000) else None
        top_pe_unwind_stk = max_pe_unwind['strike'] if (max_pe_unwind and max_pe_unwind['put_delta'] < -5000) else None

        # Detailed breakdown of Call vs Put flow concentration across strikes
        call_flow = {
            'top_writing_strike': max_ce_write['strike'] if (max_ce_write and max_ce_write['call_delta'] > 0) else None,
            'top_writing_delta': max_ce_write['call_delta'] if (max_ce_write and max_ce_write['call_delta'] > 0) else 0,
            'top_writing_lots': max_ce_write['call_lots'] if (max_ce_write and max_ce_write['call_delta'] > 0) else 0,
            'top_unwinding_strike': max_ce_unwind['strike'] if (max_ce_unwind and max_ce_unwind['call_delta'] < 0) else None,
            'top_unwinding_delta': max_ce_unwind['call_delta'] if (max_ce_unwind and max_ce_unwind['call_delta'] < 0) else 0,
            'top_unwinding_lots': max_ce_unwind['call_lots'] if (max_ce_unwind and max_ce_unwind['call_delta'] < 0) else 0,
            'total_call_delta': ce_delta,
            'total_call_vel': ce_vel,
            'total_call_oi': active_res.get('total_call_oi', 0)
        }

        put_flow = {
            'top_writing_strike': max_pe_write['strike'] if (max_pe_write and max_pe_write['put_delta'] > 0) else None,
            'top_writing_delta': max_pe_write['put_delta'] if (max_pe_write and max_pe_write['put_delta'] > 0) else 0,
            'top_writing_lots': max_pe_write['put_lots'] if (max_pe_write and max_pe_write['put_delta'] > 0) else 0,
            'top_unwinding_strike': max_pe_unwind['strike'] if (max_pe_unwind and max_pe_unwind['put_delta'] < 0) else None,
            'top_unwinding_delta': max_pe_unwind['put_delta'] if (max_pe_unwind and max_pe_unwind['put_delta'] < 0) else 0,
            'top_unwinding_lots': max_pe_unwind['put_lots'] if (max_pe_unwind and max_pe_unwind['put_delta'] < 0) else 0,
            'total_put_delta': pe_delta,
            'total_put_vel': pe_vel,
            'total_put_oi': active_res.get('total_put_oi', 0)
        }

        return {
            'regime': regime,
            'bias': bias,
            'bias_color': bias_color,
            'regime_color': bias_color,
            'net_flow_per_min': net_vel,
            'net_velocity': net_vel,
            'net_delta_contracts': net_delta,
            'total_call_delta': ce_delta,
            'total_put_delta': pe_delta,
            'total_call_vel': ce_vel,
            'total_put_vel': pe_vel,
            'major_levels': major_levels,
            'hotspots': hotspots,
            'is_dual_unwind': is_dual_unwind,
            'is_dual_write': is_dual_write,
            'call_flow': call_flow,
            'put_flow': put_flow,
            'total_call_oi': active_res.get('total_call_oi', 0),
            'total_put_oi': active_res.get('total_put_oi', 0),
            'day_pcr': active_res.get('day_pcr', 1.0),
            # Dual key contracts for compatibility
            'top_call_writing_strike': top_ce_write_stk,
            'top_ce_write_strike': top_ce_write_stk,
            'top_put_writing_strike': top_pe_write_stk,
            'top_pe_write_strike': top_pe_write_stk,
            'top_call_unwinding_strike': top_ce_unwind_stk,
            'top_ce_unwind_strike': top_ce_unwind_stk,
            'top_put_unwinding_strike': top_pe_unwind_stk,
            'top_pe_unwind_strike': top_pe_unwind_stk,
            'timeframe_matrix': tf_matrix,
            'tf_matrix': tf_matrix,
            'advisory_narrative': narrative,
            'narrative': narrative
        }

    def _get_empty_analysis(self, spot: float) -> Dict[str, Any]:
        empty_matrix = {
            '1m': {'net_vel': 0, 'net_delta': 0, 'bias': 'NEUTRAL', 'bias_color': '#94a3b8'},
            '3m': {'net_vel': 0, 'net_delta': 0, 'bias': 'NEUTRAL', 'bias_color': '#94a3b8'},
            '5m': {'net_vel': 0, 'net_delta': 0, 'bias': 'NEUTRAL', 'bias_color': '#94a3b8'},
            '15m': {'net_vel': 0, 'net_delta': 0, 'bias': 'NEUTRAL', 'bias_color': '#94a3b8'}
        }
        return {
            'regime': 'INITIALIZING...',
            'bias': 'NEUTRAL',
            'bias_color': '#94a3b8',
            'regime_color': '#94a3b8',
            'net_flow_per_min': 0,
            'net_velocity': 0,
            'net_delta_contracts': 0,
            'total_call_delta': 0,
            'total_put_delta': 0,
            'total_call_vel': 0,
            'total_put_vel': 0,
            'major_levels': {
                'call_wall': {'strike': None, 'status': 'INITIALIZING', 'status_color': '#94a3b8'},
                'put_wall': {'strike': None, 'status': 'INITIALIZING', 'status_color': '#94a3b8'},
                'atm': {'strike': spot, 'status': 'INITIALIZING', 'status_color': '#94a3b8'}
            },
            'hotspots': [],
            'top_call_writing_strike': None,
            'top_ce_write_strike': None,
            'top_put_writing_strike': None,
            'top_pe_write_strike': None,
            'top_call_unwinding_strike': None,
            'top_ce_unwind_strike': None,
            'top_put_unwinding_strike': None,
            'top_pe_unwind_strike': None,
            'timeframe_matrix': empty_matrix,
            'tf_matrix': empty_matrix,
            'advisory_narrative': 'Waiting for consecutive option chain ticks to compute Open Interest shifts...',
            'narrative': 'Waiting for consecutive option chain ticks to compute Open Interest shifts...'
        }
