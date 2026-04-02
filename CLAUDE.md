# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Living document**: As project priorities shift, update this file to reflect current goals, constraints, and conventions. Outdated instructions are worse than none — keep them accurate.

## Project Overview

Shrubbery is an experimental ML pipeline for generating predictions for Numerai, a hedge fund that crowdsources predictions from data scientists using anonymized data. It uses GPU-accelerated models, era-aware time series processing, and Weights & Biases for experiment tracking.

## Build & Development Commands

**Package manager**: `uv` (dependencies in `pyproject.toml`, lockfile in `uv.lock`)

```bash
# Install dependencies (including dev)
uv pip install --system '.[dev]'

# Linting (matches CI)
ruff check --select I src        # Import sorting
ruff format --check              # Format check
ty check                         # Type checking

# Auto-fix formatting
ruff format
ruff check --select I --fix src
```

**Docker build & run** (primary execution method):
```bash
python run.py              # Run in Docker with GPU (CI-built Docker container image)
python run.py --lint       # Run linting only in Docker
python run.py --debug      # Run in Docker, but just bash shell
python run.py --local      # Run in Docker with GPU (local-built Docker container image)
```

There is no test suite in this project.

## Code Style

- **Line length**: 79 characters
- **Quotes**: Single quotes (`'`)
- **Formatter/linter**: ruff (configured in `pyproject.toml`)
- **Editor config**: 4-space indentation, LF line endings, UTF-8
- **Comments**: Use less comments and empty lines within code; prefer decomposition into functions with meaningful names

## Architecture

### Core Pipeline (`src/shrubbery/`)

**Entry point**: `main.py` — `NumeraiRunner` orchestrates the full pipeline: data download → feature selection → model training → tournament submission. `NumeraiBestGridSearchEstimator` wraps scikit-learn's GridSearchCV with embargo-aware cross-validation.

**Data layer** (`data/`): `ingest.py` downloads/caches Numerai datasets with SHA512 checksums. `augmentation.py` and `downsampling.py` handle data preprocessing including adversarial validation filtering.

**Model universe** (`universe/`): Model wrappers (XGBoost, LightGBM, FNN, ResNet, Wide&Deep, TPOT, Factorization Machines) following scikit-learn's estimator interface.

**Embeddings** (`embeddings/`): Feature embedding strategies — SOM (Self-Organizing Maps), Autoencoder, GAN, and a generic wrapper.

**Key abstractions**:
- `NumeraiMetaEstimator` (`meta_estimator.py`): scikit-learn compatible meta-estimator that applies prediction neutralization
- `NumeraiTimeSeriesSplitter` (`cross_validation.py`): Era-aware CV splitter with embargo periods to prevent data leakage
- `Ensembler` (`ensemble.py`): Multi-model combining via product-and-root or sum-and-rank methods
- `neutralization.py`: Reduces prediction correlation to risky features using pseudo-inverse techniques

### Key Design Patterns

- All models follow **scikit-learn's BaseEstimator interface** for interoperability
- Processing is **era-aware** — Numerai's time periods ("eras") are respected throughout splitting, evaluation, and neutralization
- **Keras uses PyTorch backend** (set via `os.environ['KERAS_BACKEND'] = 'torch'`)
- **GPU-first**: NVIDIA CUDA acceleration via cuML, cuDF, XGBoost GPU, PyTorch

### Environment Variables (`.env`)

- `NUMERAI_PUBLIC_ID`, `NUMERAI_SECRET_KEY` — Numerai API credentials
- `NUMERAI_MODEL` — model name for submissions
- `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT` — Weights & Biases tracking

## CI/CD

GitHub Actions (`.github/workflows/ci.yaml`): On every push, builds a Docker image to GHCR (`ghcr.io/{owner}/shrubbery`), runs linting inside the container, and prunes old images (keeps latest 10).
