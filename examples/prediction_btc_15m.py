import os
import sys
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

CACHE_FILE = "./data/btcusdt_15m_cache.json"
CACHE_TTL_SECONDS = 60 * 15  # refresh after one full candle
LOOKBACK = 400
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def fetch_binance_klines(symbol="BTCUSDT", interval="15m", limit=400):
    resp = requests.get(BINANCE_KLINES_URL, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def load_klines(force_refresh=False):
    if not force_refresh and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        age = time.time() - cached["fetched_at"]
        if age < CACHE_TTL_SECONDS:
            print(f"Using cached data ({age:.0f}s old, TTL={CACHE_TTL_SECONDS}s)")
            return cached["klines"]

    print("Fetching fresh data from Binance...")
    klines = fetch_binance_klines(limit=LOOKBACK)
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump({"fetched_at": time.time(), "klines": klines}, f)
    return klines


def parse_klines(raw):
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "amount", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(ET)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = df[col].astype(float)
    return df


if __name__ == "__main__":
    # 1. Load & parse candles
    raw = load_klines()
    df = parse_klines(raw)

    x_df = df[["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df["open_time"]

    # Next candle open time (15 min after last candle)
    next_open = df["open_time"].iloc[-1] + timedelta(minutes=15)
    y_timestamp = pd.Series([next_open])

    print(f"\nHistorical candles : {len(df)}")
    print(f"Last candle open   : {df['open_time'].iloc[-1]}")
    print(f"Predicting candle  : {next_open}")
    print(f"Last close price   : {df['close'].iloc[-1]:.2f}\n")

    # 2. Load model
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    # 3. Predict next candle
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=1,
        T=1.0,
        top_p=0.9,
        sample_count=5,
        verbose=True,
    )

    # 4. Print result
    last_close = df["close"].iloc[-1]
    pred_df["direction"] = pred_df["close"].apply(lambda c: "UP" if c > last_close else "DOWN")

    candle_start = next_open.strftime("%H:%M")
    candle_end = (next_open + timedelta(minutes=15)).strftime("%H:%M")
    pred_df.index = [f"{candle_start} - {candle_end}"]

    predicted_at = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    row = pred_df.iloc[0]
    print(f"\n[{predicted_at}]  {candle_start}-{candle_end} ET  |  {row['direction']}  |  O:{row['open']:.2f}  H:{row['high']:.2f}  L:{row['low']:.2f}  C:{row['close']:.2f}  V:{row['volume']:.2f}")
