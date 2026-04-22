"""
Download full BTCUSDT 15m history from Binance and save as CSV
ready for the finetune_csv pipeline.

Usage:
    python download_btc_15m.py                        # full history from 2017-08-17
    python download_btc_15m.py --start 2022-01-01     # from a specific date
    python download_btc_15m.py --out data/btc_15m.csv
"""

import argparse
import time
import requests
import pandas as pd
from datetime import datetime, timezone

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
LIMIT = 1000  # max per request
DEFAULT_START = "2017-08-17"
DEFAULT_OUT = "data/btc_15m.csv"


def fetch_batch(start_ms: int, end_ms: int) -> list:
    resp = requests.get(
        BINANCE_KLINES_URL,
        params={
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": LIMIT,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def download(start_date: str, out_path: str):
    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rows = []
    current_ms = start_ms
    batch_num = 0

    print(f"Downloading BTCUSDT 15m from {start_date} ...")

    while current_ms < end_ms:
        batch = fetch_batch(current_ms, end_ms)
        if not batch:
            break

        all_rows.extend(batch)
        batch_num += 1
        last_open_ms = batch[-1][0]
        last_open_dt = datetime.fromtimestamp(last_open_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"  batch {batch_num:4d} — {len(batch)} candles — up to {last_open_dt} UTC  (total: {len(all_rows)})")

        if len(batch) < LIMIT:
            break

        # next batch starts after the last open time + 15 min
        current_ms = last_open_ms + 15 * 60 * 1000
        time.sleep(0.15)  # stay well under Binance rate limit (1200 req/min)

    print(f"\nTotal candles fetched: {len(all_rows)}")

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "amount", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = df[col].astype(float)

    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    df = df.drop_duplicates("timestamps").sort_values("timestamps").reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
    print(f"Date range: {df['timestamps'].iloc[0]}  →  {df['timestamps'].iloc[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path")
    args = parser.parse_args()

    download(args.start, args.out)
