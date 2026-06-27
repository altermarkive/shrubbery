# Numerai Experiments

Shrubbery is an experimental ML pipeline for generating predictions for Numerai, a hedge fund where trades are determined based on predictions crowdsourced from data scientists given anonymized data. It uses GPU-accelerated models, era-aware time series processing, and Weights & Biases for experiment tracking.

## Living Document

This file provides guidance to the user and coding agent when working with code in this repository.

As project priorities shift, this file is meant to be updated to reflect current goals, constraints, and conventions. Outdated instructions are worse than none — keep them accurate.

## Build & Development Commands

- **Package manager**: `uv` (dependencies in `pyproject.toml`, lockfile in `uv.lock`)
- **Sandboxing**: `podman`

**Running**:

Models are executed by the user in a Docker container via shrubbery's `run.py`:

```bash
# Run model inference
run.py -- example.py

# Run model training
run.py -- example.py --retrain
```

Models are executed by the coding agent already inside the Docker container directly via `uv`:

```bash
# Run model inference
uv run python example.py

# Run model training
uv run python example.py --retrain
```

**Linting**:
```bash
/bin/sh .github/workflows/linting.sh
```

## Architecture

### Core Pipeline (`src/shrubbery/`)

**Entry point**: `main.py` - `NumeraiRunner` implements training harness for model pipeline: data download → feature selection → model training → tournament submission. `WandbGridSearchCV` wraps scikit-learn's GridSearchCV to store results on W&B and is meant fir use with embargo-aware cross-validation.

**Data layer** (`data/`): `ingest.py` downloads/caches Numerai datasets. `augmentation.py` and `downsampling.py` handle data preprocessing.

**Model universe** (`universe/`): Model wrappers (XGBoost, FNN, ResNet, Wide&Deep, TPOT, Factorization Machines) following scikit-learn's estimator interface.

**Embeddings** (`embeddings/`): Feature embedding strategies - Autoencoder, GAN, and a generic wrapper which concatenates features and embeddings.

**Key abstractions**:
- `NumeraiNeutralization` (`neutralization.py`): scikit-learn compatible meta-estimator that applies feature exposure neutralization (reduces prediction correlation to risky features using pseudo-inverse techniques)
- `NumeraiTimeSeriesSplitter` (`validation.py`): Era-aware CV splitter with embargo periods to prevent data leakage
- `CombinatorialEnsembler` (`ensemble.py`): Multi-model combining via product-and-root or sum-and-rank methods

### Key Design Patterns

- All models follow **scikit-learn's BaseEstimator interface** for interoperability
- Processing is **era-aware** - Numerai's time periods ("eras") are respected throughout splitting, evaluation, and neutralization
- **GPU-first**: NVIDIA CUDA acceleration via cuML, XGBoost GPU, PyTorch

### Simplified Model Example

You can see an example use of the package in `example.py`.

## Environment Variables (`.env`)

To run the code create `.env` script which sets the necessary environment variables:

- `NUMERAI_PUBLIC_ID` & `NUMERAI_SECRET_KEY` - Numerai API credentials
- `NUMERAI_MODEL` - name of the model for submissions of predictions 
- `WANDB_API_KEY` - credentials to the Weights & Biases API
- `WANDB_ENTITY` & `WANDB_PROJECT` - the identifiers of Weights & Biases entity & project to upload plots and tables to

## CI/CD

GitHub Actions (`.github/workflows/ci.yaml`): On every push, it runs linting, builds and pushes a Docker container image to GHCR (`ghcr.io/{owner}/shrubbery`), and prunes old images (keeps latest 10).

## Video Introduction

[![shrubbery](http://img.youtube.com/vi/93C9VbA6h1U/0.jpg)](http://www.youtube.com/watch?v=93C9VbA6h1U)

## Profiling

Instructions in this section are meant for manual use by the user.

Profiling Compute:

```shell
# From: https://developer.nvidia.com/nsight-systems/get-started
curl -fsSL https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_2/NsightSystems-linux-cli-public-2026.2.1.210-3763964.deb -o NsightSystems-linux-cli-public-2026.2.1.210-3763964.deb
sudo dpkg -i NsightSystems-linux-cli-public-2026.2.1.210-3763964.deb
nsys profile -o trace --trace=cuda,nvtx,osrt uv run src/shrubbery/example.py --retrain
nsys export --type=perfetto --output=trace.pftrace trace.nsys-rep
curl -fsSL https://raw.githubusercontent.com/chenyu-jiang/nsys2json/main/nsys2json.py -o nsys2json.py
python3 nsys2json.py -f trace.sqlite -o trace.json
# Upload to: https://ui.perfetto.dev/
```

Profiling Memory:

```shell
uv run --with memray python -m memray run -o output.bin ../numerai/ails.py --retrain
uv run --with memray python -m memray flamegraph output.bin
```
