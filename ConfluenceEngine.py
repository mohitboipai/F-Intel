"""
ConfluenceEngine.py
===================
The institutional decision engine. Evaluates multiple independent market models 
(Regime Analysis, Option Chain Sentiment, IV/RV Divergence, Skew / Term Structure)
using a weighted probability framework to produce a final, high-conviction 
Market Verdict.
"""

class ConfluenceEngine:
    def __init__(self):
        # Weights for the scoring matrix
        self.weights = {
            'regime': 0.40,
            'gex': 0.30,
            'skew': 0.20,
            'momentum': 0.10
        }

    def evaluate(self, regime_data, iv_surface_pred, seller_data, momentum_data) -> dict:
        """
        Synthesizes modular data into a Market Verdict.
        Returns a dict: {'verdict': 'BULLISH', 'confidence': 0.85, 'rationale': [...]}
        """
        if not regime_data or not iv_surface_pred:
            return {'verdict': 'NEUTRAL', 'score': 0.0, 'confidence': 0.0, 'rationale': ['Insufficient Data']}

        rationale = []

        # 1. Regime Impact (40%)
        regime_name = regime_data.get('regime', {}).get('name', 'UNKNOWN')
        bias = regime_data.get('regime', {}).get('bias', 'NEUTRAL')
        vrp_iv_rv = regime_data.get('vrp', {}).get('iv_rv', 0)
        
        regime_score = 0.0
        if bias == 'UPSIDE': regime_score += 0.8
        elif bias == 'DOWNSIDE': regime_score -= 0.8
        
        # If VRP is deeply negative, gamma is cheap, larger moves expected
        if vrp_iv_rv < -2.0:
            rationale.append(f"Deeply negative VRP ({vrp_iv_rv:+.1f}%) → Cheap Gamma")
        elif vrp_iv_rv > 2.0:
            rationale.append(f"Expensive Gamma ({vrp_iv_rv:+.1f}%) → Mean reversion likely")
            regime_score *= 0.5 # dampens conviction of breakouts

        rationale.append(f"Regime baseline ({regime_name}): {bias}")
        
        # 2. IV Surface & Skew Impact (20%)
        pred_dir = iv_surface_pred.get('direction', 'NEUTRAL')
        pred_conf = iv_surface_pred.get('confidence', 0.5)
        skew_score = 0.0
        if pred_dir == 'BULLISH':
            skew_score = pred_conf
            rationale.append(f"Call Skew indicates Institutional Buying ({pred_conf:.0%} conf)")
        elif pred_dir == 'BEARISH':
            skew_score = -pred_conf
            rationale.append(f"Put Skew indicates Downside Protection ({pred_conf:.0%} conf)")
            
        # 3. GEX / Option Chain Data (30%)
        gex_score = 0.0
        if seller_data:
            oi_p = seller_data.get('oi_pressure', 'NEUTRAL')
            oi_s = min(100, seller_data.get('oi_pressure_score', 0)) / 100.0
            if oi_p == 'BULLISH':
                gex_score += oi_s
                rationale.append(f"Heavy Put Writing → Support Base")
            elif oi_p == 'BEARISH':
                gex_score -= oi_s
                rationale.append(f"Heavy Call Writing → Resistance Ceiling")
                
        # 4. Intraday Momentum (10%)
        mom_score = 0.0
        if momentum_data:
            m_stat = momentum_data.get('status', 'NEUTRAL')
            if m_stat == 'LONG': 
                mom_score = 1.0
                rationale.append("Spot > Intraday VWAP & EMA")
            elif m_stat == 'SHORT': 
                mom_score = -1.0
                rationale.append("Spot < Intraday VWAP & EMA")
            
        # Final weighted score (-1.0 to 1.0)
        final_score = (
            regime_score * self.weights['regime'] +
            skew_score * self.weights['skew'] +
            gex_score * self.weights['gex'] +
            mom_score * self.weights['momentum']
        )
        
        # Map to Verdict
        if final_score >= 0.5: verdict = "STRONG BULLISH"
        elif final_score >= 0.15: verdict = "BULLISH"
        elif final_score <= -0.5: verdict = "STRONG BEARISH"
        elif final_score <= -0.15: verdict = "BEARISH"
        else: verdict = "NEUTRAL / RANGEBOUND"
        
        confidence = min(abs(final_score) * 1.5, 1.0) # Scale confidence up slightly
        
        return {
            'verdict': verdict,
            'score': round(final_score, 3),
            'confidence': round(confidence, 3),
            'rationale': rationale
        }
