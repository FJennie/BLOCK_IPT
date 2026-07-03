#!/usr/bin/env bash
set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
TEST_DIR="$ROOT/test/sparse/CH4"
STAMP="$(date +%Y%m%d_%H%M%S)"

export COMPARE_TAG="${COMPARE_TAG:-best_so_far_${STAMP}}"
export IPT_DAVIDSON_USE_BEST_SO_FAR="${IPT_DAVIDSON_USE_BEST_SO_FAR:-1}"

echo "===== CH4 k=30 best-so-far debug wrapper ====="
echo "COMPARE_TAG=$COMPARE_TAG"
echo "IPT_DAVIDSON_USE_BEST_SO_FAR=$IPT_DAVIDSON_USE_BEST_SO_FAR"
echo "RESULT_SUBDIR=$ROOT/results/sparse/CH4_K30_debug/ipt_vs_primme_k30_v1/$COMPARE_TAG"
echo "LOG_SUBDIR=$ROOT/logs/sparse/CH4_K30_dubug/ipt_vs_primme_k30_v1/$COMPARE_TAG"

exec bash "$TEST_DIR/run_ch4_k30_ipt_vs_primme_compare.sh"
