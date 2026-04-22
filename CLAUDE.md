# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kronos is an open-source foundation model for financial market prediction using K-line (candlestick / OHLCV) data. It uses a two-stage hierarchical discrete tokenization framework followed by autoregressive Transformer-based generation. The paper was accepted at AAAI 2026.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run inference examples:**
```bash
python examples/prediction_example.py
python examples/prediction_batch_example.py
```

**Run tests:**
```bash
pytest tests/test_kronos_regression.py
# Single parametrized test (context length 512):
pytest tests/test_kronos_regression.py::test_kronos_predictor_regression[512]
```

**Web UI:**
```bash
python webui/app.py   # serves on http://localhost:7070
```

**Fine-tune on CSV data (sequential, single or multi-GPU):**
```bash
python finetune_csv/train_sequential.py --config finetune_csv/configs/config_ali09988_candle-5min.yaml
# Distributed (8 GPUs):
DIST_BACKEND=nccl torchrun --standalone --nproc_per_node=8 finetune_csv/train_sequential.py --config <config.yaml>
```

**Fine-tune on Qlib (A-share market) data:**
```bash
python finetune/qlib_data_preprocess.py
torchrun --standalone --nproc_per_node=2 finetune/train_tokenizer.py
torchrun --standalone --nproc_per_node=2 finetune/train_predictor.py
python finetune/qlib_test.py --device cuda:0
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

- **`finetune_csv/`**: CSV-based pipeline using YAML configs. `train_sequential.py` is the main entry point and runs tokenizer training followed by predictor training. Configs live in `finetune_csv/configs/`.
- **`finetune/`**: Qlib-based pipeline for Chinese A-share data. Has separate scripts for data preprocessing, tokenizer training, predictor training, and backtesting.

### Available pretrained models (Hugging Face)

| Model | Params | Context |
|-------|--------|---------|
| Kronos-mini | 4.1M | 2048 |
| Kronos-small | 24.7M | 512 |
| Kronos-base | 102.3M | 512 |

Tokenizers: `Kronos-Tokenizer-2k` (for mini), `Kronos-Tokenizer-base` (for small/base).

### Tests

`tests/test_kronos_regression.py` contains parametrized regression tests over multiple context lengths. Tests are split into deterministic (top_k=1, top_p=1.0) and probabilistic (sampling) categories, verified via MSE against fixed expected outputs stored in `tests/data/`.
