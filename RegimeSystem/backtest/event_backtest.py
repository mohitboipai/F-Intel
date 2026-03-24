import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from RegimeSystem.data.spot_data import SpotDataManager
from RegimeSystem.data.options_data import OptionsDataManager
from RegimeSystem.features.volatility import VolatilityFeatures
from RegimeSystem.features.options import OptionsFeatures
from RegimeSystem.model.train import SEQ_LEN

MODEL_PATH = "regime_lstm_v1.keras"

class BacktestEngine:
    def __init__(self, start_capital=100000):
        self.start_capital = start_capital
        self.cost_per_turnover = 0.0005 # 0.05%
        
    def calculate_metrics(self, df_results):
        df = df_results.copy()
        df['strategy_ret'] = df['pnl'] / 100.0
        
        sharpe = (df['strategy_ret'].mean() / df['strategy_ret'].std()) * np.sqrt(252)
        
        df['cum_ret'] = (1 + df['strategy_ret']).cumprod()
        df['peak'] = df['cum_ret'].cummax()
        df['drawdown'] = (df['cum_ret'] - df['peak']) / df['peak']
        max_dd = df['drawdown'].min()
        
        wins = len(df[df['pnl'] > 0])
        total = len(df[df['pnl'] != 0])
        win_rate = wins / total if total > 0 else 0
        
        return {
            "Sharpe": sharpe,
            "MaxDD": max_dd,
            "WinRate": win_rate,
            "TotalReturn": (df['cum_ret'].iloc[-1] - 1) * 100
        }

    def run_simulation(self, meta_df, probas, strategy_func):
        results = []
        equity = 1.0
        equity_curve = [1.0]

        for i, row in meta_df.iterrows():
            prob = probas[i]
            iv = row['iv']
            
            # Get Signal from Strategy Logic
            regime = strategy_func(prob, iv)
            
            # PnL Logic
            iv_sigma = iv / 16.0 
            ret_pct = np.log(row['next_close'] / row['close']) * 100
            abs_ret = abs(ret_pct)
            
            pnl = 0.0 # In % terms
            
            if regime == 'Long Gamma':
                pnl = abs_ret - iv_sigma
            elif regime == 'Short Vol':
                pnl = iv_sigma - abs_ret
            
            if regime != 'Neutral':
                pnl -= 0.05 # Transaction Cost
            
            # Append Results
            results.append({
                'date': row['date'],
                'pnl': pnl,
                'regime': regime
            })
            
            # Update Equity (Compounding)
            equity *= (1 + pnl/100.0)
            equity_curve.append(equity)
            
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.set_index('date')
        
        return results_df, equity_curve[:-1] # Match length

# --- Strategy Definitions ---

def strategy_original(prob, iv):
    if prob > 0.55: return 'Long Gamma'
    elif prob < 0.30: return 'Short Vol'
    return 'Neutral'

def strategy_pure_short(prob, iv):
    # Only Sell Vol, never buy.
    if prob < 0.30: return 'Short Vol'
    return 'Neutral'

def strategy_pure_long(prob, iv):
    # Only Buy Vol, never sell.
    # Higher threshold for conviction
    if prob > 0.60: return 'Long Gamma'
    return 'Neutral'

def strategy_smart_filtered(prob, iv):
    # Short Vol only if IV is high enough to be worth it (> 12)
    # Long Gamma only if IV is not too expensive (< 24)
    if prob < 0.30 and iv > 12.0: return 'Short Vol'
    if prob > 0.60 and iv < 24.0: return 'Long Gamma'
    return 'Neutral'

# ----------------------------

def prepare_backtest_data():
    print("Fetching Data...")
    spot_dm = SpotDataManager()
    spot_df = spot_dm.fetch_history(days=2500) # Increased history
    
    opt_dm = OptionsDataManager()
    vix_df = opt_dm.fetch_iv_history(days=2500)
    
    if spot_df.empty or vix_df.empty:
        return None, None, None
        
    f_vol = VolatilityFeatures.add_features(spot_df)
    f_opt = OptionsFeatures.add_features(vix_df)
    
    features_df = f_vol.join(f_opt, how='inner').dropna()
    data_df = features_df.join(spot_df[['close']], how='inner')
    
    model_features = data_df.drop(columns=['close']).values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(model_features)
    
    X = []
    meta = []
    
    for i in range(len(data_scaled) - SEQ_LEN):
        X.append(data_scaled[i : i+SEQ_LEN])
        curr_idx = i + SEQ_LEN - 1
        
        if curr_idx + 1 < len(data_df):
            meta.append({
                'date': data_df.index[curr_idx],
                'next_close': data_df['close'].iloc[curr_idx+1],
                'close': data_df['close'].iloc[curr_idx],
                'iv': data_df['iv'].iloc[curr_idx]
            })
            
    return np.array(X[:len(meta)]), pd.DataFrame(meta), scaler

def run_backtest():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        return

    X, meta_df, _ = prepare_backtest_data()
    if X is None: return
    
    print(f"Data Loaded. {len(X)} days.")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    probas = model.predict(X, verbose=0).flatten()
    
    engine = BacktestEngine()
    
    strategies = {
        "Original": strategy_original,
        "Pure Short": strategy_pure_short,
        "Pure Long": strategy_pure_long,
        "Smart Filtered": strategy_smart_filtered
    }
    
    plt.figure(figsize=(12, 6))
    
    print("\n=== Multi-Strategy Comparison ===")
    print(f"{'Strategy':<15} | {'Sharpe':<8} | {'MaxDD':<8} | {'Return':<8} | {'WinRate':<8}")
    print("-" * 65)
    
    for name, func in strategies.items():
        results_df, equity_curve = engine.run_simulation(meta_df, probas, func)
        metrics = engine.calculate_metrics(results_df)
        
        print(f"{name:<15} | {metrics['Sharpe']:<8.2f} | {metrics['MaxDD']:<8.2%} | {metrics['TotalReturn']:<8.1f}% | {metrics['WinRate']:<8.1%}")
        
        # Plot
        # Re-construct indices for plotting
        if not results_df.empty:
             plt.plot(results_df.index, equity_curve, label=f"{name} (Sharpe: {metrics['Sharpe']:.2f})")
    
    plt.title("Multi-Strategy Regimes Backtest (Equity Curves)")
    plt.xlabel("Date")
    plt.ylabel("Equity (Normalized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("multi_strategy_results.png")
    print("\nSaved comparison to 'multi_strategy_results.png'")

if __name__ == "__main__":
    run_backtest()
