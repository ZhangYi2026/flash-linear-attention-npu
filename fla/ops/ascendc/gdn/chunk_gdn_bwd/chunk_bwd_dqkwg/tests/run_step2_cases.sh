#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CACHE_ROOT="${DQKWG_STEP2_CACHE_ROOT:-$PWD/.step2_cache_seed20242617}"
RESULTS_DIR="${DQKWG_STEP2_RESULTS_DIR:-$PWD/step2_results}"
DEVICE_ID="${TEST_DEVICE_ID:-14}"
WARMUP="${DQKWG_STEP2_WARMUP:-3}"
REPEAT="${DQKWG_STEP2_REPEAT:-20}"

runner_args=()
if [ "${DQKWG_STEP2_REFRESH_CACHE:-0}" = "1" ]; then
  runner_args+=(--refresh-cache)
fi
if [ "${DQKWG_STEP2_SKIP_GOLDEN:-0}" = "1" ]; then
  runner_args+=(--skip-golden)
fi
if [ "${DQKWG_STEP2_SKIP_PERF:-0}" = "1" ]; then
  runner_args+=(--skip-perf)
fi

python3 -u step2_runner.py \
  --cache-root "$CACHE_ROOT" \
  --results-dir "$RESULTS_DIR" \
  --cpu-ref "$PWD/step2_cpu_ref.py" \
  --device "$DEVICE_ID" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  "${runner_args[@]}" \
  "$@"
