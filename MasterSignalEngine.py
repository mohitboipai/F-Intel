"""
MasterSignalEngine.py
=====================
The supreme overarching decision logic for the F-Intel platform.
Replaces the old weighted ConfluenceEngine with a factual Decision Matrix that 
cross-references Gamma, Vanna/Charm, Regime (VRP), and Intraday Momentum.
"""

class MasterSignalEngine:
    def __init__(self):
        pass

    def evaluate(self, regime_data, iv_surface_pred, seller_data, momentum_data, gex_data, dealer_data) -> dict:
        """
        Generates the absolute final verdict using the Decision Matrix.
        """
        if not regime_data or not gex_data or not dealer_data:
            return {'verdict': 'NEUTRAL', 'score': 0.0, 'confidence': 0.0, 'rationale': ['Insufficient Data for Synthesis']}

        rationale = []
        
        # 1. Extract Core Telemetry
        # -------------------------
        # Regime & VRP
        regime_name = regime_data.get('regime', {}).get('name', 'UNKNOWN')
        vrp_iv_rv   = regime_data.get('vrp', {}).get('iv_rv', 0)
        
        # Gamma & Flows
        net_gex   = gex_data.get('net_gex', 0)
        net_vanna = dealer_data.get('net_vanna', 0)
        net_charm = dealer_data.get('net_charm', 0)
        
        # Momentum
        m_stat = momentum_data.get('status', 'NEUTRAL') if momentum_data else 'NEUTRAL'

        # 2. Structural Analysis Matrix
        # -----------------------------
        is_long_gamma = net_gex > 0
        is_short_gamma = net_gex < 0
        
        is_vrp_expanding = vrp_iv_rv < -1.0 # IV is structurally cheap, expanding
        is_vrp_crushing  = vrp_iv_rv > 2.0  # IV is structurally expensive, crushing
        
        bullish_flows = (net_vanna > 0 and net_charm > 0)
        bearish_flows = (net_vanna < 0 and net_charm < 0)
        
        score = 0.0
        
        # --- A. GAMMA vs MOMENTUM CORRELATION ---
        if is_long_gamma:
            rationale.append("Long Gamma Regime: Dealers are suppressing momentum. Mean reversion expected.")
            if m_stat == 'LONG':
                rationale.append("⚠️ FAKE-OUT WARNING: Bullish momentum breakout into Long Gamma resistance.")
                score -= 0.2 # Fade the breakout
            elif m_stat == 'SHORT':
                rationale.append("⚠️ FAKE-OUT WARNING: Bearish momentum breakdown into Long Gamma support.")
                score += 0.2 # Fade the breakdown
        elif is_short_gamma:
            rationale.append("Short Gamma Regime: Dealers are accelerating momentum. Trend expected.")
            if m_stat == 'LONG':
                rationale.append("🔥 SYNCHRONIZED: Bullish momentum supported by Short Gamma hedging.")
                score += 0.8
            elif m_stat == 'SHORT':
                rationale.append("🔥 SYNCHRONIZED: Bearish momentum supported by Short Gamma hedging.")
                score -= 0.8
                
        # --- B. REGIME vs GREEK FLOWS CORRELATION ---
        if is_vrp_crushing and bullish_flows:
            rationale.append("STRUCTURAL BUY: Expensive Gamma (crushing IV) triggers positive Vanna buying.")
            score += 0.6
        elif is_vrp_expanding and bearish_flows:
            rationale.append("STRUCTURAL SELL: Expanding IV triggers negative Vanna selling.")
            score -= 0.6

        # --- C. OPTION ANALYTICS (OI) FILTER ---
        if seller_data:
            oi_p = seller_data.get('oi_pressure', 'NEUTRAL')
            if oi_p == 'BULLISH' and score >= 0:
                rationale.append("Retail OI Put Writing aligns with bullish bias.")
                score += 0.3
            elif oi_p == 'BEARISH' and score <= 0:
                rationale.append("Retail OI Call Writing aligns with bearish bias.")
                score -= 0.3

        # 3. Final Verdict Mapping
        # ------------------------
        if score >= 1.0: verdict = "STRONG BULLISH"
        elif score > 0.3: verdict = "BULLISH"
        elif score <= -1.0: verdict = "STRONG BEARISH"
        elif score < -0.3: verdict = "BEARISH"
        else: verdict = "NEUTRAL / RANGEBOUND"
        
        # If deeply Long Gamma, we cap conviction because everything reverts
        if is_long_gamma and abs(score) > 0.5:
            rationale.append("Conviction capped due to Long Gamma pinning effects.")
            confidence = 0.5
        else:
            confidence = min(abs(score), 1.0)
            
        return {
            'verdict': verdict,
            'score': round(score, 3),
            'confidence': round(confidence, 3),
            'rationale': rationale
        }
