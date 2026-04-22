#!/usr/bin/env python
"""
CLI wrapper for KronosClient. Prints a single JSON result to stdout.
Called as a subprocess by other projects — do not import directly.

Usage:
    python kronos_cli.py [--candle_time 2026-04-20T14:45] [--model auto|pretrained|finetuned]
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kronos_client import KronosClient

parser = argparse.ArgumentParser()
parser.add_argument("--candle_time", default=None, help="ET datetime YYYY-MM-DDTHH:MM (omit for next live candle)")
parser.add_argument("--model", default="auto", choices=["auto", "pretrained", "finetuned"])
parser.add_argument("--sample_count", type=int, default=5)
args = parser.parse_args()

client = KronosClient()
result = client.predict(
    candle_time=args.candle_time,
    model_key=args.model,
    sample_count=args.sample_count,
    verbose=False,
)
print(json.dumps(result))
