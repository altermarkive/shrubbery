# Numerai Experiments

Numerai is a hedge fund where trades are determined based on predictions crowdsourced from data scientists given anonymized data.
This repo contains my experimental code for generation of Numerai predictions.

To run the code create `.env` script which sets the necessary environment variables:

- `NUMERAI_PUBLIC_ID` & `NUMERAI_SECRET_KEY` - credentials to Numerai's API
- `NUMERAI_MODEL` - name of the model to upload predictions to
- `WANDB_API_KEY` - credentials to the Weights & Biases API
- `WANDB_ENTITY` & `WANDB_PROJECT` - the identifiers of Weights & Biases entity & project to upload plots and tables to

You can see an example use of the package in `example.py`.

[![shrubbery](http://img.youtube.com/vi/93C9VbA6h1U/0.jpg)](http://www.youtube.com/watch?v=93C9VbA6h1U)

Running in `ghcr.io/altermarkive/shrubbery:latest` container:

```python
uv sync
uv pip install -e .
uv run src/shrubbery/example.py --retrain
```

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