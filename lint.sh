#!/bin/sh

SELF=$0
BASE=$(realpath $(dirname "$SELF"))
. $BASE/auth.sh

COMMAND="isort --check --diff src; black --check --diff src; flake8 src; mypy src"

docker run \
  --rm \
  -it \
  -v $PWD:/w -w /w \
  --pull=always \
  --entrypoint /bin/sh \
  ghcr.io/altermarkive/shrubbery:latest \
  -c "$COMMAND"
