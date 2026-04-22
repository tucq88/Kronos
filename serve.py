import os
import sys
import time
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from flask import Flask, jsonify, request

from model import Kronos, KronosTokenizer, KronosPredictor

# --- Config ---
API_SECRET = os.environ.get("API_SECRET", "kronos-secret-change-me")
PORT = int(os.environ.get("PORT", 8000))
LOOKBACK = 400
CACHE_TTL_SECONDS = 60 * 15
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
ET = ZoneInfo("America/New_York")

FINETUNED_TOKENIZER = os.path.join(os.path.dirname(__file__), "finetune_csv/finetuned/btc_15m/tokenizer/best_model")
FINETUNED_PREDICTOR = os.path.join(os.path.dirname(__file__), "finetune_csv/finetuned/btc_15m/basemodel/best_model")

app = Flask(__name__)

# --- Model registry (lazy-loaded, one instance per key) ---
_model_lock = threading.Lock()
_predictors: dict = {}   # key: "pretrained" | "finetuned"


def load_predictor(key: str) -> KronosPredictor:
    with _model_lock:
        if key in _predictors:
            return _predictors[key]

        if key == "finetuned":
            if not (os.path.isdir(FINETUNED_TOKENIZER) and os.path.isdir(FINETUNED_PREDICTOR)):
                raise FileNotFoundError("Fine-tuned model not found. Run the fine-tuning pipeline first.")
            print("Loading fine-tuned model ...")
            tokenizer = KronosTokenizer.from_pretrained(FINETUNED_TOKENIZER)
            model = Kronos.from_pretrained(FINETUNED_PREDICTOR)
        else:
            print("Loading pretrained Kronos-small ...")
            tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

        predictor = KronosPredictor(model, tokenizer, max_context=512)
        _predictors[key] = predictor
        print(f"Model ready: {key}")
        return predictor


def default_model_key() -> str:
    return "finetuned" if (os.path.isdir(FINETUNED_TOKENIZER) and os.path.isdir(FINETUNED_PREDICTOR)) else "pretrained"


# --- Binance data ---
def fetch_klines(end_time_ms: int = None) -> list:
    params = {"symbol": "BTCUSDT", "interval": "15m", "limit": LOOKBACK}
    if end_time_ms:
        params["endTime"] = end_time_ms
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# Simple cache for current (live) candles only
_cache_lock = threading.Lock()
_live_cache = {"klines": None, "fetched_at": 0}


def get_live_klines():
    with _cache_lock:
        age = time.time() - _live_cache["fetched_at"]
        if _live_cache["klines"] is not None and age < CACHE_TTL_SECONDS:
            return _live_cache["klines"], age
        klines = fetch_klines()
        _live_cache["klines"] = klines
        _live_cache["fetched_at"] = time.time()
        return klines, 0


def parse_klines(raw) -> pd.DataFrame:
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


# --- Auth ---
def require_auth(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {API_SECRET}":
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


# --- Routes ---
@app.route("/predict/btc-15m", methods=["GET"])
@require_auth
def predict_btc_15m():
    try:
        # Model selection
        model_key = request.args.get("model", default_model_key())
        if model_key not in ("pretrained", "finetuned"):
            return jsonify({"error": "model must be 'pretrained' or 'finetuned'"}), 400

        # Optional historical candle_time (ET, e.g. 2026-04-20T14:45)
        candle_time_str = request.args.get("candle_time")
        is_historical = candle_time_str is not None

        if is_historical:
            candle_open = datetime.fromisoformat(candle_time_str).replace(tzinfo=ET)
            # Fetch LOOKBACK candles ending just before candle_open
            end_ms = int(candle_open.timestamp() * 1000) - 1
            raw = fetch_klines(end_time_ms=end_ms)
            cache_age = None
        else:
            raw, cache_age = get_live_klines()

        df = parse_klines(raw)
        if len(df) < 2:
            return jsonify({"error": "Not enough historical data returned"}), 500

        x_df = df[["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp = df["open_time"]
        last_close = float(df["close"].iloc[-1])

        if is_historical:
            next_open = candle_open
        else:
            next_open = df["open_time"].iloc[-1] + timedelta(minutes=15)

        y_timestamp = pd.Series([next_open])

        predictor = load_predictor(model_key)
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

        pred_close = round(float(pred_df["close"].iloc[0]), 2)
        direction = "UP" if pred_close > last_close else "DOWN"

        result = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "model": f"Kronos-small / {model_key}",
            "candle_open": next_open.strftime("%Y-%m-%d %H:%M ET"),
            "direction": direction,
            "last_close": round(last_close, 2),
            "predicted_open":  round(float(pred_df["open"].iloc[0]), 2),
            "predicted_high":  round(float(pred_df["high"].iloc[0]), 2),
            "predicted_low":   round(float(pred_df["low"].iloc[0]), 2),
            "predicted_close": pred_close,
        }

        if cache_age is not None:
            result["cache_age_seconds"] = round(cache_age)

        # For historical requests, fetch the actual candle for scoring
        if is_historical:
            actual_end_ms = int((candle_open + timedelta(minutes=15)).timestamp() * 1000)
            actual_raw = fetch_klines(end_time_ms=actual_end_ms)
            actual_df = parse_klines(actual_raw)
            actual_candle = actual_df[actual_df["open_time"] == candle_open]
            if not actual_candle.empty:
                actual_close = round(float(actual_candle["close"].iloc[0]), 2)
                actual_direction = "UP" if actual_close > last_close else "DOWN"
                result["actual_close"] = actual_close
                result["actual_direction"] = actual_direction
                result["correct"] = direction == actual_direction

        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/models", methods=["GET"])
@require_auth
def models():
    finetuned_available = os.path.isdir(FINETUNED_TOKENIZER) and os.path.isdir(FINETUNED_PREDICTOR)
    return jsonify({
        "default": default_model_key(),
        "pretrained": {"available": True, "loaded": "pretrained" in _predictors},
        "finetuned": {"available": finetuned_available, "loaded": "finetuned" in _predictors},
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Warm up default model before accepting traffic
    load_predictor(default_model_key())
    app.run(host="0.0.0.0", port=PORT)
