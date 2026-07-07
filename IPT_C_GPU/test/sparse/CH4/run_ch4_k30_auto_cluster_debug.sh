#!/usr/bin/env bash
#SBATCH --job-name=ch4_k30_auto
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/ch4_k30_auto_cluster_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/ch4_k30_auto_cluster_%j.err

set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
TEST_DIR="$ROOT/test/sparse/CH4"
STAMP="$(date +%Y%m%d_%H%M%S)"

export COMPARE_TAG="${COMPARE_TAG:-cluster_fix_auto_${STAMP}}"
export IPT_COMPARE_FORCE_ACTIVE_PAIRS=""
export IPT_DAVIDSON_CONVERGED_TOL=1e-12
export IPT_DAVIDSON_USE_BEST_SO_FAR=1
export IPT_DAVIDSON_RELAXED_ACCEPT=1
export IPT_DAVIDSON_RETRY_ON_REJECT=1
export IPT_DAVIDSON_MIN_ACCEPTED_STEPS="${IPT_DAVIDSON_MIN_ACCEPTED_STEPS:-12}"
export IPT_DAVIDSON_CLUSTER_AWARE_ACCEPT=1
export IPT_DAVIDSON_SOFT_CLUSTER_LOCKING=1
export IPT_DAVIDSON_LOCKED_DEGRADE_ABS_SLACK=1e-12
export IPT_DAVIDSON_RESTART_EVERY=0

unset IPT_DAVIDSON_FORCE_ACTIVE_PAIRS
unset IPT_DAVIDSON_ACTIVE_CLUSTERS
unset IPT_DAVIDSON_ACTIVE_TOL
unset IPT_DAVIDSON_ACTIVE_MAX
unset IPT_DAVIDSON_CORRECTION_MAX_PER_STEP
unset IPT_DAVIDSON_DEBUG_ACTIVE_MAX

echo "===== CH4 k=30 automatic Ritz-cluster Davidson ====="
echo "COMPARE_TAG=$COMPARE_TAG"
echo "partition=${SLURM_JOB_PARTITION:-gpu}"
echo "automatic_cluster_discovery=1"
echo "converged_tol=$IPT_DAVIDSON_CONVERGED_TOL"
echo "force_active_pairs_set=0"
echo "manual_active_clusters_set=0"
echo "active_tol_used=0"
echo "active_max_used=0"
echo "correction_max_per_step_used=0"
echo "restart_every=$IPT_DAVIDSON_RESTART_EVERY"
echo "RESULT_SUBDIR=$ROOT/results/sparse/CH4_K30_debug/ipt_vs_primme_k30_v1/$COMPARE_TAG"
echo "LOG_SUBDIR=$ROOT/logs/sparse/CH4_K30_dubug/ipt_vs_primme_k30_v1/$COMPARE_TAG"

exec bash "$TEST_DIR/run_ch4_k30_ipt_vs_primme_compare.sh"
