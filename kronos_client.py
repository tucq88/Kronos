"""
Importable client for BTC 15m candle prediction.

Usage from any local project:
    import sys
    sys.path.insert(0, "/path/to/Kronos")
    from kronos_client import KronosClient

    client = KronosClient()  # loads model once
    result = client.predict()                              # next candle
    result = client.predict(candle_time="2026-04-20T14:45")  # historical
    result = client.predict(model_key="finetuned")

Result dict:
    {
        "candle_open":      "2026-04-22 10:00 ET",
        "direction":        "UP" | "DOWN",
        "model":            "Kronos-small / pretrained",
        "last_close":       93241.50,
        "predicted_open":   93280.12,
        "predicted_high":   93450.00,
        "predicted_low":    93100.00,
        "predicted_close":  93390.25,
        # historical only:
        "actual_close":     93150.00,
        "actual_direction": "DOWN",
        "correct":          False,
    }
"""

import os
import time
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd

ET = ZoneInfo("America/New_York")
LOOKBACK = 400
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
CACHE_TTL_SECONDS = 60 * 15

_KRONOS_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_TOKENIZER = os.path.join(_KRONOS_DIR, "pretrained/Kronos-Tokenizer-base")
PRETRAINED_PREDICTOR = os.path.join(_KRONOS_DIR, "pretrained/Kronos-small")
FINETUNED_TOKENIZER  = os.path.join(_KRONOS_DIR, "finetune_csv/finetuned/btc_15m/tokenizer/best_model")
FINETUNED_PREDICTOR  = os.path.join(_KRONOS_DIR, "finetune_csv/finetuned/btc_15m/basemodel/best_model")


def _import_model():
    import sys
    sys.path.insert(0, _KRONOS_DIR)
    from model import Kronos, KronosTokenizer, KronosPredictor
    return Kronos, KronosTokenizer, KronosPredictor


def _fetch_klines(end_time_ms: int = None) -> list:
    params = {"symbol": "BTCUSDT", "interval": "15m", "limit": LOOKBACK}
    if end_time_ms:
        params["endTime"] = end_time_ms
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _parse_klines(raw) -> pd.DataFrame:
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


def _finetuned_available() -> bool:
    return os.path.isdir(FINETUNED_TOKENIZER) and os.path.isdir(FINETUNED_PREDICTOR)


class KronosClient:
    def __init__(self, default_model: str = "auto"):
        """
        Args:
            default_model: "auto" picks finetuned if available else pretrained,
                           or explicitly "pretrained" / "finetuned"
        """
        self._lock = threading.Lock()
        self._predictors: dict = {}
        self._live_cache = {"klines": None, "fetched_at": 0}
        self._cache_lock = threading.Lock()
        self.default_model = default_model

    def _resolve_model_key(self, model_key: str) -> str:
        if model_key == "auto":
            return "finetuned" if _finetuned_available() else "pretrained"
        return model_key

    def _load_predictor(self, key: str):
        with self._lock:
            if key in self._predictors:
                return self._predictors[key]

            Kronos, KronosTokenizer, KronosPredictor = _import_model()

            if key == "finetuned":
                if not _finetuned_available():
                    raise FileNotFoundError("Fine-tuned model not found. Run the fine-tuning pipeline first.")
                print("Loading fine-tuned model ...")
                tokenizer = KronosTokenizer.from_pretrained(FINETUNED_TOKENIZER)
                model = Kronos.from_pretrained(FINETUNED_PREDICTOR)
            else:
                src_tok = PRETRAINED_TOKENIZER if os.path.isdir(PRETRAINED_TOKENIZER) else "NeoQuasar/Kronos-Tokenizer-base"
                src_mdl = PRETRAINED_PREDICTOR if os.path.isdir(PRETRAINED_PREDICTOR) else "NeoQuasar/Kronos-small"
                print(f"Loading pretrained Kronos-small from {src_mdl} ...")
                tokenizer = KronosTokenizer.from_pretrained(src_tok)
                model = Kronos.from_pretrained(src_mdl)

            from model import KronosPredictor as KP
            predictor = KP(model, tokenizer, max_context=512)
            self._predictors[key] = predictor
            print(f"Model ready: {key}")
            return predictor

    def _get_live_klines(self):
        with self._cache_lock:
            age = time.time() - self._live_cache["fetched_at"]
            if self._live_cache["klines"] is not None and age < CACHE_TTL_SECONDS:
                return self._live_cache["klines"]
            klines = _fetch_klines()
            self._live_cache["klines"] = klines
            self._live_cache["fetched_at"] = time.time()
            return klines

    def predict(self, candle_time: str = None, model_key: str = "auto",
                sample_count: int = 5, verbose: bool = False) -> dict:
        """
        Args:
            candle_time: ET datetime string "YYYY-MM-DDTHH:MM" for historical,
                         None for next live candle.
            model_key:   "auto" | "pretrained" | "finetuned"
            sample_count: number of forward passes to average (higher = more stable)
            verbose:     show inference progress bar
        """
        key = self._resolve_model_key(model_key)
        is_historical = candle_time is not None

        if is_historical:
            candle_open = datetime.fromisoformat(candle_time).replace(tzinfo=ET)
            end_ms = int(candle_open.timestamp() * 1000) - 1
            raw = _fetch_klines(end_time_ms=end_ms)
        else:
            raw = self._get_live_klines()

        df = _parse_klines(raw)
        x_df = df[["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp = df["open_time"]
        last_close = float(df["close"].iloc[-1])

        next_open = candle_open if is_historical else df["open_time"].iloc[-1] + timedelta(minutes=15)
        y_timestamp = pd.Series([next_open])

        predictor = self._load_predictor(key)
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=1,
            T=1.0,
            top_p=0.9,
            sample_count=sample_count,
            verbose=verbose,
        )

        pred_close = round(float(pred_df["close"].iloc[0]), 2)
        direction = "UP" if pred_close > last_close else "DOWN"

        result = {
            "candle_open":     next_open.strftime("%Y-%m-%d %H:%M ET"),
            "direction":       direction,
            "model":           f"Kronos-small / {key}",
            "last_close":      round(last_close, 2),
            "predicted_open":  round(float(pred_df["open"].iloc[0]), 2),
            "predicted_high":  round(float(pred_df["high"].iloc[0]), 2),
            "predicted_low":   round(float(pred_df["low"].iloc[0]), 2),
            "predicted_close": pred_close,
        }

        if is_historical:
            actual_end_ms = int((candle_open + timedelta(minutes=15)).timestamp() * 1000)
            actual_raw = _fetch_klines(end_time_ms=actual_end_ms)
            actual_df = _parse_klines(actual_raw)
            actual_candle = actual_df[actual_df["open_time"] == candle_open]
            if not actual_candle.empty:
                actual_close = round(float(actual_candle["close"].iloc[0]), 2)
                actual_direction = "UP" if actual_close > last_close else "DOWN"
                result["actual_close"] = actual_close
                result["actual_direction"] = actual_direction
                result["correct"] = direction == actual_direction

        return result
