# Numerai Experiments

Numerai is a hedge fund where trades are determined based on predictions crowdsourced from data scientists given anonymized data.
This repo contains my experimental code for generation of Numerai predictions.

To run the code create `.env` script which sets the necessary environment variables:

- `NUMERAI_PUBLIC_ID` & `NUMERAI_SECRET_KEY` - credentials to Numerai's API
- `NUMERAI_MODEL` - name of the model to upload predictions to
- `WANDB_API_KEY` - credentials to the Weights & Biases API
- `WANDB_ENTITY` & `WANDB_PROJECT` - the identifiers of Weights & Biases entity & project to upload plots and tables to

Once that is ready run the code by using the `run.py` script.

[![shrubbery](http://img.youtube.com/vi/93C9VbA6h1U/0.jpg)](http://www.youtube.com/watch?v=93C9VbA6h1U)
