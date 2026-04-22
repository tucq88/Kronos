"""
Drop this file into any project to get BTC 15m predictions from Kronos
without installing Kronos dependencies.

Usage:
    from kronos_bridge import predict_btc

    result = predict_btc()                                 # next candle
    result = predict_btc(candle_time="2026-04-20T14:45")   # historical
    result = predict_btc(model="finetuned")

    print(result["direction"])   # "UP" or "DOWN"
    print(result["correct"])     # True/False (historical only)
"""
import json
import subprocess

KRONOS_DIR = "/Users/tucq/code/_try/Kronos"
KRONOS_PYTHON = f"{KRONOS_DIR}/.venv/bin/python"
KRONOS_CLI = f"{KRONOS_DIR}/kronos_cli.py"


def predict_btc(candle_time: str = None, model: str = "auto", sample_count: int = 5) -> dict:
    cmd = [KRONOS_PYTHON, KRONOS_CLI, "--model", model, "--sample_count", str(sample_count)]
    if candle_time:
        cmd += ["--candle_time", candle_time]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return json.loads(out)
