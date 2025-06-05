#!/bin/sh

docker run \
  --rm \
  -it \
  --gpus all \
  -v $PWD:/w -w /w \
  -e NUMERAI_PUBLIC_ID -e NUMERAI_SECRET_KEY -e NUMERAI_MODEL -e WANDB_API_KEY -e WANDB_ENTITY -e WANDB_PROJECT -e HYDRA_FULL_ERROR=1 \
  --pull=always \
  --entrypoint /bin/bash \
  ghcr.io/altermarkive/shrubbery:latest
