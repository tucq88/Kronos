# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kronos is an open-source foundation model for financial market prediction using K-line (candlestick / OHLCV) data. It uses a two-stage hierarchical discrete tokenization framework followed by autoregressive Transformer-based generation. The paper was accepted at AAAI 2026.

This fork extends the original with a BTC 15m prediction pipeline: a REST API (`serve.py`), an importable Python client (`kronos_client.py`), fine-tuning on BTCUSDT 15m data, and RunPod + VPS deployment scripts.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**BTC 15m prediction (CLI):**
```bash
python examples/prediction_btc_15m.py
```

**Run the API server locally:**
```bash
API_SECRET=your-secret python serve.py   # http://localhost:8000
```

**Run tests:**
```bash
pytest tests/test_kronos_regression.py
pytest tests/test_kronos_regression.py::test_kronos_predictor_regression[512]
```

**Web UI:**
```bash
python webui/app.py   # serves on http://localhost:7070
```

**Fine-tune on BTC 15m data:**
```bash
# Step 1: download historical data (~220k candles from 2020)
python finetune_csv/download_btc_15m.py --start 2020-01-01 --out finetune_csv/data/btc_15m.csv

# Step 2: train (from finetune_csv/)
python train_sequential.py --config configs/config_btc_15m.yaml

# Resume from a checkpoint (e.g. after accidental stop at epoch 3/20)
python train_sequential.py --config configs/config_btc_15m_resume.yaml --skip-tokenizer
```

**Update VPS:**
```bash
./scripts/vps_update.sh root@<vps-ip> -i ~/.ssh/your-key
```

## Model layout

All models live under the repo root — no reliance on HuggingFace cache:

```
pretrained/                              ← downloaded once, gitignored
├── Kronos-Tokenizer-base/
└── Kronos-small/
finetune_csv/finetuned/btc_15m/          ← produced by fine-tuning, gitignored
├── tokenizer/best_model/
└── basemodel/best_model/
```

`serve.py` and `kronos_client.py` load from local paths first, fall back to HuggingFace if missing. Fine-tuned model is used by default when present.

To download pretrained models locally:
```python
from huggingface_hub import snapshot_download
snapshot_download('NeoQuasar/Kronos-Tokenizer-base', local_dir='pretrained/Kronos-Tokenizer-base')
snapshot_download('NeoQuasar/Kronos-small', local_dir='pretrained/Kronos-small')
```

## BTC 15m prediction API (`serve.py`)

Single Flask process serving both models. Auth via `Authorization: Bearer <secret>`.

```bash
# Next candle (fine-tuned by default)
curl http://localhost:8000/predict/btc-15m -H "Authorization: Bearer <secret>"

# Force model
curl "http://localhost:8000/predict/btc-15m?model=pretrained"

# Historical candle — returns actual_close / actual_direction / correct for backtesting
curl "http://localhost:8000/predict/btc-15m?model=finetuned&candle_time=2026-04-20T14:45"

# Check which models are loaded
curl http://localhost:8000/models -H "Authorization: Bearer <secret>"
```

**VPS:** `http://104.248.94.217:8000` — managed by systemd (`systemctl status kronos`), logs at `/var/log/kronos/`.

## Importable Python client (`kronos_client.py` / `kronos_bridge.py`)

For other local projects that need predictions without HTTP overhead:

**Option A — direct import** (requires Kronos deps in the calling project):
```python
import sys
sys.path.insert(0, "/Users/tucq/code/_try/Kronos")
from kronos_client import KronosClient

client = KronosClient()                                        # loads model once
result = client.predict()                                      # next candle
result = client.predict(candle_time="2026-04-20T14:45")        # historical
result = client.predict(model_key="pretrained", sample_count=10)
```

**Option B — subprocess bridge** (zero deps in calling project, copy `kronos_bridge.py` there):
```python
from kronos_bridge import predict_btc

result = predict_btc()
result = predict_btc(candle_time="2026-04-20T14:45", model="finetuned")
print(result["direction"], result.get("correct"))
```

`kronos_bridge.py` calls `kronos_cli.py` via Kronos's own `.venv` — no dep bleed.

### Response fields

| Field | When | Description |
|---|---|---|
| `direction` | always | `"UP"` or `"DOWN"` vs previous close |
| `model` | always | e.g. `"Kronos-small / finetuned"` |
| `candle_open` | always | e.g. `"2026-04-22 10:00 ET"` |
| `last_close` | always | Baseline close used for direction |
| `predicted_open/high/low/close` | always | Full predicted OHLC |
| `actual_close` / `actual_direction` | historical | Real values from Binance |
| `correct` | historical | `True/False` direction accuracy |

## RunPod fine-tuning workflow

```bash
# 1. Push code + data to pod
./scripts/push_to_runpod.sh "user@ssh.runpod.io -i ~/.ssh/your-key" --with-data

# 2. On the pod
bash /workspace/Kronos/scripts/runpod_setup.sh
bash /workspace/Kronos/scripts/runpod_train.sh   # or GPUS=2 for multi-GPU

# 3. Pull model back
./scripts/pull_from_runpod.sh "user@ssh.runpod.io -i ~/.ssh/your-key"

# 4. Push to VPS
rsync -avz -e "ssh -i ~/.ssh/your-key" \
  finetune_csv/finetuned/btc_15m/ root@<vps>:/opt/kronos/finetune_csv/finetuned/btc_15m/
ssh -i ~/.ssh/your-key root@<vps> "systemctl restart kronos"
```

## Architecture

### Core components (`model/`)

The public API is exported from `model/__init__.py`: `KronosTokenizer`, `Kronos`, `KronosPredictor`.

**`model/kronos.py`** — three main classes:

- **`KronosTokenizer`**: Encodes continuous 6D OHLCV data into hierarchical discrete token indices (s1 = coarse, s2 = fine) using an encoder-decoder Transformer + `BinarySphericalQuantizer`. Supports `encode()` and `decode()`.
- **`Kronos`**: Autoregressive Transformer with hierarchical embeddings. Predicts s1 tokens, then predicts s2 tokens conditioned on the corresponding s1 token via a dependency-aware cross-attention layer. Uses Rotary Position Embeddings (RoPE) and dual prediction heads.
- **`KronosPredictor`**: High-level end-to-end API. Handles normalization → tokenization → iterative autoregressive inference → decoding → denormalization. Returns pandas DataFrames. `predict()` supports repeated sampling + averaging for uncertainty quantification; `predict_batch()` parallelizes across multiple inputs.

**`model/module.py`** — neural network primitives:
- `BinarySphericalQuantizer`: Entropy-regularized VQ with binary codes.
- `MultiHeadAttentionWithRoPE`: Standard MHA with rotary embeddings.
- `HierarchicalEmbedding`: Separate embedding tables for s1/s2 token types.
- `TemporalEmbedding`: Encodes minute/hour/weekday/day/month from timestamps.
- `DualHead`: Parallel prediction heads for s1 and s2 tokens.

### Data flow

```
DataFrame [open,high,low,close,volume,amount] + timestamps
  → normalize (mean/std, clip ±5)
  → KronosTokenizer.encode() → (s1_indices, s2_indices)
  → Kronos.decode_s1() / decode_s2()  [autoregressive loop]
  → KronosTokenizer.decode() → reconstructed features
  → denormalize
  → DataFrame (forecast)
```

### Fine-tuning pipelines

- **`finetune_csv/`**: CSV-based pipeline using YAML configs. `train_sequential.py` is the main entry point. BTC configs: `config_btc_15m.yaml` (fresh run), `config_btc_15m_resume.yaml` (resume from checkpoint).
- **`finetune/`**: Qlib-based pipeline for Chinese A-share data.

### Pretrained models (Hugging Face)

| Model | Params | Context |
|-------|--------|---------|
| Kronos-mini | 4.1M | 2048 |
| Kronos-small | 24.7M | 512 |
| Kronos-base | 102.3M | 512 |

Tokenizers: `Kronos-Tokenizer-2k` (for mini), `Kronos-Tokenizer-base` (for small/base).

### Tests

`tests/test_kronos_regression.py` — parametrized regression tests over multiple context lengths, deterministic (top_k=1) and probabilistic categories, verified via MSE against fixed expected outputs in `tests/data/`.
