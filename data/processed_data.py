import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

RAW_DIR       = r"raw"         # relative — works if script is inside data/
PROCESSED_DIR = r"processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── INDICATOR FUNCTIONS ──────────────────────────────────────────────────────

def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast      = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow      = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"]    = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    return df

def add_bollinger(df, period=20):
    ma              = df["Close"].rolling(period).mean()
    std             = df["Close"].rolling(period).std()
    df["BB_Upper"]  = ma + 2 * std
    df["BB_Lower"]  = ma - 2 * std
    df["BB_Width"]  = df["BB_Upper"] - df["BB_Lower"]          # volatility
    df["BB_Pct"]    = (df["Close"] - df["BB_Lower"]) / df["BB_Width"]  # position within band
    return df

def add_ema(df):
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_30"] = df["Close"].ewm(span=30, adjust=False).mean()
    df["EMA_Cross"] = df["EMA_10"] - df["EMA_30"]   # positive = bullish crossover
    return df

def add_atr(df, period=14):
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"]  = tr.ewm(com=period - 1, min_periods=period).mean()
    return df

def add_obv(df):
    direction  = df["Close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["OBV"]  = (df["Volume"] * direction).cumsum()
    return df

def add_stochastic(df, k_period=14, d_period=3):
    low_min      = df["Low"].rolling(k_period).min()
    high_max     = df["High"].rolling(k_period).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
    df["Stoch_D"] = df["Stoch_K"].rolling(d_period).mean()
    return df

def add_roc(df, period=10):
    df["ROC"] = df["Close"].pct_change(periods=period) * 100
    return df

def add_volume_ma_ratio(df, period=20):
    df["Vol_MA_Ratio"] = df["Volume"] / df["Volume"].rolling(period).mean()
    return df

def add_all_indicators(df):
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_ema(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_stochastic(df)
    df = add_roc(df)
    df = add_volume_ma_ratio(df)
    return df

# ── COLUMNS TO USE AS FEATURES ───────────────────────────────────────────────
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",   # raw OHLCV
    "RSI",                                        # momentum
    "MACD", "MACD_Signal", "MACD_Hist",          # trend
    "BB_Width", "BB_Pct",                         # volatility / price position
    "EMA_10", "EMA_30", "EMA_Cross",             # trend direction
    "ATR",                                        # volatility
    "OBV",                                        # volume-price momentum
    "Stoch_K", "Stoch_D",                        # overbought/oversold
    "ROC",                                        # rate of change
    "Vol_MA_Ratio",                              # unusual volume
]

# ── PROCESS EACH STOCK ───────────────────────────────────────────────────────
for fname in os.listdir(RAW_DIR):
    if not fname.endswith("_raw.csv"):
        continue

    symbol = fname.replace("_raw.csv", "")
    df = pd.read_csv(os.path.join(RAW_DIR, fname), parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset="Date")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Add all indicators
    df = add_all_indicators(df)

    # Drop rows where indicators are NaN (warm-up period)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Normalize all feature columns to [0, 1]
    scaler = MinMaxScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    # Final output: Date + Symbol + all features
    out = df[["Date", "Symbol"] + FEATURE_COLS]
    out.to_csv(os.path.join(PROCESSED_DIR, f"{symbol}_processed.csv"), index=False)
    print(f" {symbol}  {len(out)} rows  |  {len(FEATURE_COLS)} features")

print("\n All stocks processed!")