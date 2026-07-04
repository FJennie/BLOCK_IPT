#!/usr/bin/env bash
#SBATCH --job-name=ch4_k30_relax
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/ch4_k30_relaxed_accept_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/ch4_k30_relaxed_accept_%j.err

set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
TEST_DIR="$ROOT/test/sparse/CH4"
STAMP="$(date +%Y%m%d_%H%M%S)"

export COMPARE_TAG="${COMPARE_TAG:-relaxed_accept_${STAMP}}"
export IPT_DAVIDSON_USE_BEST_SO_FAR="${IPT_DAVIDSON_USE_BEST_SO_FAR:-1}"
export IPT_DAVIDSON_RELAXED_ACCEPT="${IPT_DAVIDSON_RELAXED_ACCEPT:-1}"
export IPT_DAVIDSON_ACCEPT_REL_SLACK="${IPT_DAVIDSON_ACCEPT_REL_SLACK:-1e-12}"
export IPT_DAVIDSON_ACCEPT_ABS_SLACK="${IPT_DAVIDSON_ACCEPT_ABS_SLACK:-1e-15}"
export IPT_DAVIDSON_ACTIVE_PAIR_ACCEPT="${IPT_DAVIDSON_ACTIVE_PAIR_ACCEPT:-1}"
export IPT_DAVIDSON_RETRY_ON_REJECT="${IPT_DAVIDSON_RETRY_ON_REJECT:-1}"
export IPT_DAVIDSON_RETRY_DAMPING_LIST="${IPT_DAVIDSON_RETRY_DAMPING_LIST:-0.5,0.25}"
export IPT_DAVIDSON_RETRY_DENOM_CLIP_MULTS="${IPT_DAVIDSON_RETRY_DENOM_CLIP_MULTS:-10,100}"
export IPT_DAVIDSON_MIN_ACCEPTED_STEPS="${IPT_DAVIDSON_MIN_ACCEPTED_STEPS:-12}"

echo "===== CH4 k=30 relaxed-accept debug wrapper ====="
echo "COMPARE_TAG=$COMPARE_TAG"
echo "IPT_DAVIDSON_USE_BEST_SO_FAR=$IPT_DAVIDSON_USE_BEST_SO_FAR"
echo "IPT_DAVIDSON_RELAXED_ACCEPT=$IPT_DAVIDSON_RELAXED_ACCEPT"
echo "IPT_DAVIDSON_RETRY_ON_REJECT=$IPT_DAVIDSON_RETRY_ON_REJECT"
echo "IPT_DAVIDSON_MIN_ACCEPTED_STEPS=$IPT_DAVIDSON_MIN_ACCEPTED_STEPS"
echo "RESULT_SUBDIR=$ROOT/results/sparse/CH4_K30_debug/ipt_vs_primme_k30_v1/$COMPARE_TAG"
echo "LOG_SUBDIR=$ROOT/logs/sparse/CH4_K30_dubug/ipt_vs_primme_k30_v1/$COMPARE_TAG"

exec bash "$TEST_DIR/run_ch4_k30_ipt_vs_primme_compare.sh"
