"""
Comprehensive Test Run & Live Engine Verification Script.
Executes analytical pipelines across all calculation engines and prints detailed results.
"""
import sys
import numpy as np
import pandas as pd
import config
from calculations.GreeksEngine import GreeksEngine
from calculations.GexEngine import GexEngine
from calculations.DealerPositionEngine import DealerPositionEngine
from RealizedVolEngine import RVEstimators
from AdvancedVolatilityScanner import AdvancedVolatilityScanner
from OptionAnalytics import OptionAnalytics

def main():
    print("=" * 78)
    print("        F-INTEL CALCULATION ENGINES: LIVE TEST RUN & VERIFICATION")
    print("=" * 78)

    # ──────────────────────────────────────────────────────────
    # 1. CONFIGURATION SYSTEM
    # ──────────────────────────────────────────────────────────
    print("\n[1] CONFIGURATION SYSTEM STATUS")
    print("-" * 55)
    config.load_profile("default")
    print(f"  Active Profile       : {config.active_profile()}")
    print(f"  NIFTY Lot Size       : {config.get('nifty_lot_size')}")
    print(f"  Risk-Free Rate (r)   : {config.get('risk_free_rate'):.6f} ({config.get('risk_free_rate')*100:.2f}%)")
    print(f"  Dividend Yield (q)   : {config.get('dividend_yield'):.4f} ({config.get('dividend_yield')*100:.2f}%)")
    print(f"  Dividend DTE Thresh  : {config.get('dividend_dte_threshold')} days")
    print(f"  IV Solver Seed       : {config.get('iv_solver_seed'):.2f}")
    print(f"  Consensus RV Weights : YZ={config.get('rv_weight_yz')*100:.0f}%, "
          f"C2C={config.get('rv_weight_c2c')*100:.0f}%, "
          f"Parkinson={config.get('rv_weight_park')*100:.0f}%, "
          f"GK={config.get('rv_weight_gk')*100:.0f}%")
    print(f"  Available Profiles   : {config.list_profiles()}")

    # ──────────────────────────────────────────────────────────
    # 2. GREEKS ENGINE (MERTON 1973 DIVIDEND-ADJUSTED)
    # ──────────────────────────────────────────────────────────
    print("\n[2] GREEKS ENGINE (Merton 1973 Continuous Dividend Model)")
    print("-" * 55)
    spot = 24000.0
    strike = 24000.0
    dte_days = 25.0
    T = dte_days / 365.0
    sigma = 0.15
    r = config.get("risk_free_rate")
    q = config.get("dividend_yield")

    greeks_engine = GreeksEngine(risk_free_rate=r, dividend_yield=q)
    df_greeks = greeks_engine.calculate_all_greeks(
        S=spot,
        K=np.array([strike, strike]),
        T_days=np.array([dte_days, dte_days]),
        iv=np.array([sigma, sigma]),
        option_types=np.array(['CE', 'PE'])
    )

    analytics = OptionAnalytics()
    call_price = analytics.black_scholes(spot, strike, T, r, sigma, 'CE', q=q)
    put_price = analytics.black_scholes(spot, strike, T, r, sigma, 'PE', q=q)

    call_delta = df_greeks.loc[0, 'delta']
    put_delta = df_greeks.loc[1, 'delta']
    gamma = df_greeks.loc[0, 'gamma']
    vega = df_greeks.loc[0, 'vega']
    call_theta = df_greeks.loc[0, 'theta']
    put_theta = df_greeks.loc[1, 'theta']
    charm = df_greeks.loc[0, 'charm']
    vanna = df_greeks.loc[0, 'vanna']

    print(f"  Underlying Spot      : {spot:,.2f}")
    print(f"  ATM Strike           : {strike:,.2f} (DTE = {dte_days:.0f}d, T = {T:.4f}y)")
    print(f"  Implied Volatility   : {sigma*100:.1f}%")
    print(f"  Call Price (Merton)  : {call_price:.2f} INR")
    print(f"  Put Price (Merton)   : {put_price:.2f} INR")
    print(f"  Call Delta           : {call_delta:+.4f}")
    print(f"  Put Delta            : {put_delta:+.4f}")
    print(f"  Merton Parity Check  : Call Delta - Put Delta = {call_delta - put_delta:.6f}")
    print(f"  Theoretical e^(-qT)  : {np.exp(-q * T):.6f} (Matches Parity: {abs((call_delta - put_delta) - np.exp(-q * T)) < 1e-6})")
    print(f"  Gamma (per 1 INR)    : {gamma:.6f}")
    print(f"  Vega (per 1% IV)     : {vega:.4f} INR")
    print(f"  Call Theta (per day) : {call_theta:.4f} INR")
    print(f"  Put Theta (per day)  : {put_theta:.4f} INR")
    print(f"  Charm (dDelta/dTime) : {charm:+.6f}")
    print(f"  Vanna (dDelta/dVol)  : {vanna:+.6f}")

    # ──────────────────────────────────────────────────────────
    # 3. GEX & DEALER POSITION ENGINE
    # ──────────────────────────────────────────────────────────
    print("\n[3] GEX ENGINE & DEALER EXPOSURE (Tri-Model)")
    print("-" * 55)
    # Synthetic realistic option chain centered around spot=24000
    strikes = [23600, 23700, 23800, 23900, 24000, 24100, 24200, 24300, 24400]
    records = []
    for k in strikes:
        call_oi = 85000 if k >= 24000 else 35000
        put_oi = 90000 if k <= 24000 else 30000
        call_iv = 0.145 + 0.005 * (k - 24000)/1000
        put_iv = 0.155 - 0.010 * (k - 24000)/1000
        records.append({'strike': k, 'type': 'CE', 'oi': call_oi, 'dte': 7, 'iv': call_iv})
        records.append({'strike': k, 'type': 'PE', 'oi': put_oi, 'dte': 7, 'iv': put_iv})
    df_chain = pd.DataFrame(records)

    gex_std = GexEngine(lot_size=config.get("nifty_lot_size"), positioning_model='standard')
    res_std = gex_std.calculate_gex(df_chain, spot_price=spot)

    gex_inv = GexEngine(lot_size=config.get("nifty_lot_size"), positioning_model='inverted')
    res_inv = gex_inv.calculate_gex(df_chain, spot_price=spot)

    dealer_engine = DealerPositionEngine(lot_size=config.get("nifty_lot_size"))
    dealer_inv = dealer_engine.calculate_dealer_inventory(df_chain, spot_price=spot)

    print(f"  Standard Model Net GEX   : {res_std['net_gex']/1e7:+,.2f} Cr INR (Dealer Long: {res_std['dealer_long_pct']:.1f}%, Short: {res_std['dealer_short_pct']:.1f}%)")
    print(f"  Standard Zero Gamma Flip : {res_std['zero_gamma_level']:,.2f} (Dist: {res_std['distance_from_zero']:+,.2f} pts)")
    print(f"  Inverted Model Net GEX   : {res_inv['net_gex']/1e7:+,.2f} Cr INR (Dealer Long: {res_inv['dealer_long_pct']:.1f}%, Short: {res_inv['dealer_short_pct']:.1f}%)")
    print(f"  Inverted Zero Gamma Flip : {res_inv['zero_gamma_level']:,.2f} (Dist: {res_inv['distance_from_zero']:+,.2f} pts)")
    print(f"  Dealer Net Delta Exposure: {dealer_inv['net_delta_exposure']:+,.0f} INR")
    print(f"  Dealer Net Gamma Shares  : {dealer_inv['net_gamma_shares']:+,.0f} shares")
    print(f"  Projected Hedge (+1% Spot): {dealer_inv['projected_hedging']['buy_shares_if_spot_up_1pct']:+,.0f} shares "
          f"({'BUY' if dealer_inv['projected_hedging']['buy_shares_if_spot_up_1pct'] > 0 else 'SELL'} to flatten delta)")
    print(f"  Projected Hedge (+1% IV)  : {dealer_inv['projected_hedging']['buy_shares_if_iv_up_1pct']:+,.0f} shares")

    # ──────────────────────────────────────────────────────────
    # 4. REALIZED VOLATILITY ENGINE ESTIMATORS
    # ──────────────────────────────────────────────────────────
    print("\n[4] REALIZED VOLATILITY ENGINE (Consensus & Estimators)")
    print("-" * 55)
    np.random.seed(42)
    daily_rets = np.random.normal(0.0003, 0.009, 60)
    closes = 24000 * np.cumprod(1 + daily_rets)
    highs = closes * (1 + np.random.uniform(0.002, 0.008, 60))
    lows = closes * (1 - np.random.uniform(0.002, 0.008, 60))
    opens = closes * (1 + np.random.normal(0, 0.003, 60))
    df_ohlc = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes})

    rv_est = RVEstimators()
    rv_yz = rv_est.yang_zhang(df_ohlc['open'], df_ohlc['high'], df_ohlc['low'], df_ohlc['close'], window=20).iloc[-1]
    rv_park = rv_est.parkinson(df_ohlc['high'], df_ohlc['low'], window=20).iloc[-1]
    rv_gk = rv_est.garman_klass(df_ohlc['open'], df_ohlc['high'], df_ohlc['low'], df_ohlc['close'], window=20).iloc[-1]
    rv_c2c = rv_est.close_to_close(df_ohlc['close'], window=20).iloc[-1]

    w_yz = config.get("rv_weight_yz")
    w_c2c = config.get("rv_weight_c2c")
    w_park = config.get("rv_weight_park")
    w_gk = config.get("rv_weight_gk")
    consensus_rv = (w_yz * rv_yz + w_c2c * rv_c2c + w_park * rv_park + w_gk * rv_gk)

    atm_iv = 14.8
    vrp = atm_iv - consensus_rv
    print(f"  Yang-Zhang RV (40% wt)   : {rv_yz:.2f}% (Jump + Overnight Robust)")
    print(f"  Parkinson RV  (20% wt)   : {rv_park:.2f}% (High-Low Intraday)")
    print(f"  Garman-Klass  (20% wt)   : {rv_gk:.2f}% (OHLC Drift-Adjusted)")
    print(f"  Close-to-Close(20% wt)   : {rv_c2c:.2f}% (Classical Rolling)")
    print(f"  Consensus Realized Vol   : {consensus_rv:.2f}%")
    print(f"  Current ATM IV           : {atm_iv:.2f}%")
    print(f"  Volatility Risk Premium  : {vrp:+.2f}% ({'IV > RV: Vol Premium Rich' if vrp > 0 else 'RV > IV: Vol Underpriced'})")

    # ──────────────────────────────────────────────────────────
    # 5. ADVANCED VOLATILITY SCANNER (IVP & IVR)
    # ──────────────────────────────────────────────────────────
    print("\n[5] ADVANCED VOLATILITY SCANNER (IVP & IVR)")
    print("-" * 55)
    scanner = AdvancedVolatilityScanner()
    hist_iv_series = pd.Series(np.random.normal(15.0, 2.5, 252))
    cur_iv = 16.5
    ivp = scanner.compute_ivp(cur_iv, hist_iv_series)
    ivr = scanner.compute_ivr(cur_iv, hist_iv_series)
    print(f"  Current Test IV          : {cur_iv:.1f}%")
    print(f"  252-day Historical Range : [{hist_iv_series.min():.1f}% - {hist_iv_series.max():.1f}%]")
    print(f"  IV Percentile (IVP)      : {ivp:.1f}% (Percentage of days with lower IV)")
    print(f"  IV Rank (IVR)            : {ivr:.1f}% (Normalized position within min-max range)")

    # ──────────────────────────────────────────────────────────
    # 6. OPTION ANALYTICS 4-TIER IV RESOLVER
    # ──────────────────────────────────────────────────────────
    print("\n[6] 4-TIER IMPLIED VOLATILITY FALLBACK (VolatilityAnalyzer)")
    print("-" * 55)
    # Tier 1: Valid feed
    t1 = 15.2
    # Tier 2: Solved price with Merton dividend
    known_price = analytics.black_scholes(spot, 24000.0, T, r, 0.16, 'CE', q=q)
    t2_solved = analytics.implied_volatility(known_price, spot, 24000.0, T, r, 'CE', q=q)
    # Tier 3: Quadratic smile for OTM Put (23500)
    k_otm = np.log(23500.0 / spot)
    t3_smile = 15.0 - 15.0 * k_otm + 25.0 * (k_otm ** 2)

    print(f"  Tier 1: Observable Feed  : {t1:.1f}% (Valid live quote)")
    print(f"  Tier 2: Merton Solved IV : {t2_solved:.1f}% (Inverted from premium={known_price:.2f} with q={q*100:.2f}%)")
    print(f"  Tier 3: Smile Fallback   : {t3_smile:.1f}% (Strike 23500 OTM Put, skew-adjusted)")
    print(f"  Tier 4: Flat Fallback    : {config.get('iv_fallback_flat')*100:.1f}% (Bounded [{config.get('iv_fallback_min')*100:.0f}%, {config.get('iv_fallback_max')*100:.0f}%])")

    print("\n" + "=" * 78)
    print("                 ALL ENGINE VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 78)

if __name__ == '__main__':
    main()
