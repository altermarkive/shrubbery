#!/bin/sh

docker run \
  --rm \
  -it \
  -v $PWD:/w -w /w \
  --pull=always \
  --entrypoint /bin/sh \
  ghcr.io/altermarkive/shrubbery:latest \
  -c "isort --check --diff src; black --check --diff src; flake8 src; mypy src"
