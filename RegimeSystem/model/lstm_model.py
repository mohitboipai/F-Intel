import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, BatchNormalization, Bidirectional
)
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_shape, n_classes=3):
    """
    Builds the LSTM Regime Classifier for intraday-aware signals.

    Input Shape: (SEQ_LEN, n_features)
    Output:
        If n_classes == 3: softmax → P(LOW_VOL), P(NORMAL), P(EXPANSION)
        If n_classes == 1: sigmoid → binary (legacy)

    Architecture: 3-layer LSTM with BatchNorm + residual-style design for
    faster feature extraction — reduces SEQ_LEN to 10 for intraday reaction.
    """
    model = Sequential([
        Input(shape=input_shape),

        # Layer 1 — broad pattern extraction
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.25),

        # Layer 2 — medium-term context
        LSTM(32, return_sequences=True),
        BatchNormalization(),
        Dropout(0.20),

        # Layer 3 — final sequence encoding
        LSTM(16, return_sequences=False),
        Dropout(0.15),

        # Classification head
        Dense(32, activation='relu'),
        Dropout(0.15),
        Dense(16, activation='relu'),
    ])

    if n_classes == 1:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['AUC', 'accuracy', tf.keras.metrics.Precision(name='precision')]
        )
    else:
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

    return model


def get_intraday_bias(pred_probs: list, iv_zscore: float, iv_velocity: float) -> dict:
    """
    Derive an actionable intraday bias from model output + live IV metrics.

    Args:
        pred_probs: Softmax probs for [LOW_VOL, NORMAL, EXPANSION] (3-class)
        iv_zscore:  Current IV z-score vs 20-day mean
        iv_velocity: IV change since previous close

    Returns:
        dict with 'signal', 'confidence', 'reasoning'
    """
    if len(pred_probs) != 3:
        return {'signal': 'NEUTRAL', 'confidence': 0.0, 'reasoning': 'Insufficient model output'}

    p_low, p_normal, p_exp = pred_probs
    dominant_class = ['LOW_VOL', 'NORMAL', 'EXPANSION'][int(pred_probs.index(max(pred_probs)))]
    confidence = max(pred_probs)

    # IV velocity override: if IV is spiking hard, elevate to EXPANSION
    if iv_velocity > 1.5:
        signal = 'BUY-PREMIUM'
        reason = f'IV spiking +{iv_velocity:.1f}% → volatility expansion in progress'
    elif iv_velocity < -1.5:
        signal = 'SELL-PREMIUM'
        reason = f'IV falling {iv_velocity:.1f}% + model: {dominant_class} → sell premium conditions'
    elif dominant_class == 'EXPANSION' and p_exp > 0.55:
        signal = 'BUY-PREMIUM'
        reason = f'LSTM: P(Expansion)={p_exp:.0%} → elevated vol risk, avoid naked short premium'
    elif dominant_class in ('LOW_VOL', 'NORMAL') and iv_zscore < 0.5:
        signal = 'SELL-PREMIUM'
        reason = f'LSTM: P(Low/Normal)={p_low+p_normal:.0%} + IV z-score={iv_zscore:.1f}x → premium seller edge'
    else:
        signal = 'NEUTRAL'
        reason = f'Mixed signals — LSTM: {dominant_class} ({confidence:.0%}), IV velocity: {iv_velocity:+.1f}%'

    return {
        'signal': signal,
        'confidence': confidence,
        'reasoning': reason,
        'p_low_vol': p_low,
        'p_normal': p_normal,
        'p_expansion': p_exp,
    }
