#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --locked
uv run --directory src/lib/external python setup.py build_ext --inplace
uv run --directory src/tools/voc_eval_lib python setup.py build_ext --inplace
uv run python -m compileall -q src
uv run pytest "$@"
