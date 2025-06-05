#!/bin/sh

SELF=$0
BASE=$(realpath $(dirname "$SELF"))
. $BASE/auth.sh

COMMAND="/bin/bash"

docker run \
  --rm \
  -it \
  --gpus all \
  -v $PWD:/w -w /w \
  -e PIP_INDEX_URL -e NUMERAI_PUBLIC_ID -e NUMERAI_SECRET_KEY -e NUMERAI_MODEL -e WANDB_API_KEY -e WANDB_ENTITY -e WANDB_PROJECT -e HYDRA_FULL_ERROR=1 \
  --pull=always \
  --entrypoint /bin/sh \
  ghcr.io/altermarkive/shrubbery:latest \
  -c "$COMMAND"
