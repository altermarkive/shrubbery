#!/bin/sh

set -e

uv run --with ruff ruff check --select I src
uv run --with ruff ruff format --check --diff
uv run --with ty ty check
uv run --with deptry deptry . --ignore DEP001,DEP002
