import os
import sys
import json
import time
import threading
import requests
import pandas as pd
from datetime import timedelta
from zoneinfo import ZoneInfo
from flask import Flask, jsonify, request

from model import Kronos, KronosTokenizer, KronosPredictor

# --- Config ---
API_SECRET = os.environ.get("API_SECRET", "kronos-secret-change-me")
PORT = 8000
CACHE_TTL_SECONDS = 60 * 15
LOOKBACK = 400
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
ET = ZoneInfo("America/New_York")

# Fine-tuned model paths (produced by finetune_csv pipeline).
# Falls back to pretrained HF models when these paths don't exist.
FINETUNED_TOKENIZER = "finetune_csv/finetuned/btc_15m/tokenizer/best_model"
FINETUNED_PREDICTOR = "finetune_csv/finetuned/btc_15m/basemodel/best_model"

app = Flask(__name__)

# --- Shared state ---
_model_lock = threading.Lock()
_predictor: KronosPredictor = None

_cache_lock = threading.Lock()
_cache = {"klines": None, "fetched_at": 0}
_model_name = None


def get_predictor():
    global _predictor, _model_name
    if _predictor is None:
        with _model_lock:
            if _predictor is None:
                if os.path.isdir(FINETUNED_TOKENIZER) and os.path.isdir(FINETUNED_PREDICTOR):
                    print(f"Loading fine-tuned model from {FINETUNED_PREDICTOR} ...")
                    tokenizer = KronosTokenizer.from_pretrained(FINETUNED_TOKENIZER)
                    model = Kronos.from_pretrained(FINETUNED_PREDICTOR)
                    _model_name = "Kronos-small / fine-tuned"
                else:
                    print("Fine-tuned model not found, loading pretrained Kronos-small ...")
                    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
                    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
                    _model_name = "Kronos-small / pretrained"
                _predictor = KronosPredictor(model, tokenizer, max_context=512)
                print(f"Model ready: {_model_name}")
    return _predictor


def get_klines():
    with _cache_lock:
        age = time.time() - _cache["fetched_at"]
        if _cache["klines"] is not None and age < CACHE_TTL_SECONDS:
            return _cache["klines"], age
        resp = requests.get(
            BINANCE_KLINES_URL,
            params={"symbol": "BTCUSDT", "interval": "15m", "limit": LOOKBACK},
            timeout=10,
        )
        resp.raise_for_status()
        _cache["klines"] = resp.json()
        _cache["fetched_at"] = time.time()
        return _cache["klines"], 0


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


def require_auth(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {API_SECRET}":
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


@app.route("/predict/btc-15m", methods=["GET"])
@require_auth
def predict_btc_15m():
    try:
        raw, cache_age = get_klines()
        df = parse_klines(raw)

        x_df = df[["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp = df["open_time"]

        last_close = df["close"].iloc[-1]
        last_candle_time = df["open_time"].iloc[-1]
        next_open = last_candle_time + timedelta(minutes=15)
        y_timestamp = pd.Series([next_open])

        predictor = get_predictor()
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=1,
            T=1.0,
            top_p=0.9,
            sample_count=5,
            verbose=False,
        )

        pred_close = float(pred_df["close"].iloc[0])
        direction = "UP" if pred_close > last_close else "DOWN"

        return jsonify({
            "symbol": "BTCUSDT",
            "interval": "15m",
            "candle_open": str(next_open),
            "direction": direction,
            "last_close": round(last_close, 2),
            "predicted_open": round(float(pred_df["open"].iloc[0]), 2),
            "predicted_high": round(float(pred_df["high"].iloc[0]), 2),
            "predicted_low": round(float(pred_df["low"].iloc[0]), 2),
            "predicted_close": round(pred_close, 2),
            "cache_age_seconds": round(cache_age),
            "model": _model_name,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model", methods=["GET"])
def model_info():
    get_predictor()
    return jsonify({"model": _model_name})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Warm up model before serving
    get_predictor()
    app.run(host="0.0.0.0", port=PORT)
