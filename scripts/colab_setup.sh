#!/usr/bin/env bash
set -euo pipefail

# Minimal bootstrap for running the pipeline in Colab.
# Usage: bash scripts/colab_setup.sh

python -m pip install --upgrade pip

# Core deps and CLI from this repo (editable install assumes you mounted/cloned the repo).
python -m pip install -e .
