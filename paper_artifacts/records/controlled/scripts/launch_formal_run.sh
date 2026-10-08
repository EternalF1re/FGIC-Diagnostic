#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="./wsl_python_deps:.:./scripts"
export HF_HOME="<LOCAL_PATH> Face/模型数据/10118/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p formal_run

exec <LOCAL_PATH> scripts/formal_orchestrator.py \
  > formal_run/orchestrator.stdout.log \
  2> formal_run/orchestrator.stderr.log
