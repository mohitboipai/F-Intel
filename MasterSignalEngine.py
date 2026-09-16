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

    def evaluate(self, memory) -> dict:
        """
        Generates the absolute final verdict using the Decision Matrix.
        Pulls all required data directly from SignalMemory context.
        Also reads intraday_signal context written by IntradayGammaSignalEngine.
        """
        context = memory.get_context() if memory else {}
        if not context:
            return {'verdict': 'NEUTRAL', 'score': 0.0, 'confidence': 0.0, 'rationale': ['Insufficient Data for Synthesis']}

        rationale = []
        
        # 1. Extract Core Telemetry
        # -------------------------
        # Regime & VRP
        regime_name = context.get('regime', 'UNKNOWN')
        vrp_iv_rv   = context.get('vrp', 0)
        
        # Gamma & Flows
        net_gex   = context.get('net_gex', 0)
        net_vanna = context.get('net_vanna', 0)
        net_charm = context.get('net_charm', 0)
        
        # Momentum — prefer IntradayGammaSignalEngine reading if available
        m_stat = context.get('momentum_status', context.get('intraday_momentum', 'NEUTRAL'))

        # Intraday signal context (written by IntradayGammaSignalEngine each cycle)
        intraday_ctx    = context.get('intraday_signal', {})
        swing_quality   = intraday_ctx.get('swing_quality', {}).get('quality', 'UNKNOWN')
        session_phase   = intraday_ctx.get('swing_quality', {}).get('session_phase', 'UNKNOWN')
        intraday_score  = float(intraday_ctx.get('score', 0.0))

        # 2. Swing Quality Gate (must run before scoring)
        # ------------------------------------------------
        if swing_quality == 'CHOPPY':
            rationale.append('⚠️ CHOPPY DAY — conviction capped. Avoid directional trades.')
            return {
                'verdict':    'AVOID',
                'score':      0.0,
                'confidence': 0.0,
                'rationale':  rationale
            }

        # 3. Structural Analysis Matrix
        # -----------------------------
        is_long_gamma  = net_gex > 0
        is_short_gamma = net_gex < 0
        
        is_vrp_expanding = vrp_iv_rv < -1.0  # IV cheap, expanding
        is_vrp_crushing  = vrp_iv_rv > 2.0   # IV expensive, crushing
        
        bullish_flows = (net_vanna > 0 and net_charm > 0)
        bearish_flows = (net_vanna < 0 and net_charm < 0)
        
        score = 0.0
        
        # --- A. GAMMA vs MOMENTUM CORRELATION ---
        if is_long_gamma:
            rationale.append("Long Gamma Regime: Dealers are suppressing momentum. Mean reversion expected.")
            if m_stat == 'LONG':
                rationale.append("⚠️ FAKE-OUT WARNING: Bullish momentum breakout into Long Gamma resistance.")
                score -= 0.2  # Fade the breakout
            elif m_stat == 'SHORT':
                rationale.append("⚠️ FAKE-OUT WARNING: Bearish momentum breakdown into Long Gamma support.")
                score += 0.2  # Fade the breakdown
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
        oi_p = context.get('oi_pressure', 'NEUTRAL')
        if oi_p == 'BULLISH' and score >= 0:
            rationale.append("Retail OI Put Writing aligns with bullish bias.")
            score += 0.3
        elif oi_p == 'BEARISH' and score <= 0:
            rationale.append("Retail OI Call Writing aligns with bearish bias.")
            score -= 0.3

        # --- D. INTRADAY FUSION SIGNAL BOOST ---
        if intraday_score >= 65.0:
            boost = 0.2 if score >= 0 else -0.2
            score += boost
            rationale.append(f"Intraday signal score {intraday_score:.0f} — conviction {'boosted' if boost > 0 else 'confirmed bearish'}.")

        # --- E. SESSION PHASE MODIFIER ---
        if session_phase == 'MID_TREND':
            # Best swing window: boost final confidence 10%
            rationale.append("Mid-trend session (11:00–14:00): highest reliability window.")
        elif session_phase == 'OPENING_RANGE':
            score *= 0.8   # Reduce conviction in first 45 min
            rationale.append("Opening range caution: score reduced 20%.")

        # 4. Final Verdict Mapping
        # ------------------------
        if score >= 1.0:   verdict = "STRONG BULLISH"
        elif score > 0.3:  verdict = "BULLISH"
        elif score <= -1.0: verdict = "STRONG BEARISH"
        elif score < -0.3: verdict = "BEARISH"
        else:              verdict = "NEUTRAL / RANGEBOUND"
        
        # If deeply Long Gamma, cap conviction
        if is_long_gamma and abs(score) > 0.5:
            rationale.append("Conviction capped due to Long Gamma pinning effects.")
            confidence = 0.5
        else:
            confidence = min(abs(score), 1.0)

        # Session phase confidence boost for mid-trend
        if session_phase == 'MID_TREND':
            confidence = min(confidence * 1.10, 1.0)
            
        result = {
            'verdict':    verdict,
            'score':      round(score, 2),
            'confidence': round(confidence, 2),
            'rationale':  rationale
        }
        
        # Write verdict back into SignalMemory
        if memory:
            memory.update_context({'confluence_verdict': result})
            
        return result
