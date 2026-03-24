import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from RegimeSystem.data.spot_data import SpotDataManager
from RegimeSystem.data.options_data import OptionsDataManager
from RegimeSystem.features.volatility import VolatilityFeatures
from RegimeSystem.features.options import OptionsFeatures
from RegimeSystem.model.lstm_model import build_lstm_model

SEQ_LEN        = 10   # Lookback 10 days (reduced from 20 for faster intraday reaction)
PREDICT_WINDOW = 5    # Next 5 days (intraday-relevant horizon)
N_CLASSES      = 3    # LOW_VOL / NORMAL / EXPANSION

# IV thresholds for 3-class target
IV_LOW_THRESH  = 10.0  # Below this → LOW_VOL
IV_HIGH_THRESH = 15.0  # Above this → EXPANSION

MODEL_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'regime_lstm_v2.keras'))
SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'regime_scaler_v2.pkl'))


def create_targets(df):
    """
    3-class target based on max IV over next PREDICT_WINDOW days.
      0 = LOW_VOL    (max future IV < IV_LOW_THRESH)
      1 = NORMAL     (IV_LOW_THRESH <= max IV < IV_HIGH_THRESH)
      2 = EXPANSION  (max future IV >= IV_HIGH_THRESH)
    """
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=PREDICT_WINDOW)
    future_max_iv = df['iv'].rolling(window=indexer).max().shift(-1)

    y = pd.Series(1, index=df.index, dtype=int)  # default NORMAL
    y[future_max_iv < IV_LOW_THRESH]  = 0  # LOW_VOL
    y[future_max_iv >= IV_HIGH_THRESH] = 2  # EXPANSION
    return y


def prepare_dataset():
    # 1. Fetch
    spot_dm = SpotDataManager()
    spot_df = spot_dm.fetch_history(days=2000)

    opt_dm = OptionsDataManager()
    vix_df = opt_dm.fetch_iv_history(days=2000)

    if spot_df.empty or vix_df.empty:
        print("Error: Empty Data")
        return None, None, None

    # 2. Features (both vol and options)
    f_vol = VolatilityFeatures.add_features(spot_df)
    f_opt = OptionsFeatures.add_features(vix_df)

    # 3. Merge (inner join aligns by date)
    df = f_vol.join(f_opt, how='inner').dropna()

    # 4. Targets
    df['target'] = create_targets(df)
    valid_df = df.dropna()
    print(f"Dataset Size: {len(valid_df)} rows | Class distribution: {valid_df['target'].value_counts().to_dict()}")

    feature_cols = [c for c in valid_df.columns if c != 'target']
    data_x = valid_df[feature_cols].values
    data_y = valid_df['target'].values

    # 5. Scale
    scaler = StandardScaler()
    data_x_scaled = scaler.fit_transform(data_x)

    # 6. Build sequences
    X, y = [], []
    for i in range(len(data_x_scaled) - SEQ_LEN):
        X.append(data_x_scaled[i: i + SEQ_LEN])
        y.append(data_y[i + SEQ_LEN - 1])

    return np.array(X), np.array(y), scaler


def train():
    X, y, scaler = prepare_dataset()
    if X is None:
        return

    print(f"Training on {len(X)} sequences. Features: {X.shape[2]}")

    # Time-aware split: train first 80%, test last 20%
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Class weight (handle imbalance)
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weight = {int(c): total / (N_CLASSES * cnt) for c, cnt in zip(classes, counts)}
    print(f"Class weights: {class_weight}")

    # Build model
    model = build_lstm_model((SEQ_LEN, X.shape[2]), n_classes=N_CLASSES)
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
    ]

    # Train
    model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # Save scaler alongside model for live inference
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    # Evaluate
    from sklearn.metrics import classification_report, confusion_matrix
    preds = model.predict(X_test)
    y_pred_class = np.argmax(preds, axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_class, target_names=['LOW_VOL', 'NORMAL', 'EXPANSION']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_class))


if __name__ == "__main__":
    train()
