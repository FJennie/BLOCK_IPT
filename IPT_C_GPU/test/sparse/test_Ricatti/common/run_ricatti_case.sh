#!/usr/bin/env bash
set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
CASE_ENV="${1:?usage: run_ricatti_case.sh CASE_ENV K}"
K_VALUE="${2:?usage: run_ricatti_case.sh CASE_ENV K}"

export IPT_C_ROOT="$ROOT"
export IPT_RESULT_BASE="${IPT_RESULT_BASE:-$ROOT/results/sparse/test_Ricatti}"
export IPT_LOG_BASE="${IPT_LOG_BASE:-$ROOT/logs/sparse/test_Ricatti}"
export RUN_IPT="${RUN_IPT:-1}"
export RUN_PRIMME="${RUN_PRIMME:-0}"
export RUN_WARMUP="${RUN_WARMUP:-0}"
export IPT_REPEATS="${IPT_REPEATS:-1}"
export IPT_TOL="${IPT_TOL:-1e-12}"
export IPT_BLOCK_CLUSTER="${IPT_BLOCK_CLUSTER:-1}"
export IPT_BLOCK_CLUSTER_QR="${IPT_BLOCK_CLUSTER_QR:-1}"
export IPT_DAVIDSON_ENRICH=0
export IPT_DUMP_PAIR_RESIDUALS="${IPT_DUMP_PAIR_RESIDUALS:-1}"

bash "$ROOT/test/sparse/test_initial/common/run_initial_case.sh" \
    "$CASE_ENV" "$K_VALUE"
