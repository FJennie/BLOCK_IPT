#!/usr/bin/env bash
#SBATCH --job-name=nh3_k30_auto
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/NH3_K30/nh3_k30_auto_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/NH3_K30/nh3_k30_auto_%j.err

set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
TEST_DIR="$ROOT/test/sparse/NH3"

STAMP="$(date +%Y%m%d_%H%M%S)"
TAG="${COMPARE_TAG:-nh3_k30_auto_${STAMP}}"

export IPT_C_RESULTS_DIR="$ROOT/results/sparse/NH3_k30/$TAG"
export IPT_C_LOG_DIR="$ROOT/logs/sparse/NH3_K30/$TAG"

mkdir -p "$IPT_C_RESULTS_DIR" "$IPT_C_LOG_DIR" "$ROOT/logs/sparse/NH3_K30"

# 清掉旧的手工 active/cluster 环境变量，避免继承 CH4 的 FORCE 路径。
unset IPT_COMPARE_FORCE_ACTIVE_PAIRS
unset IPT_DAVIDSON_FORCE_ACTIVE_PAIRS
unset IPT_DAVIDSON_ACTIVE_CLUSTERS
unset IPT_DAVIDSON_ACTIVE_TOL
unset IPT_DAVIDSON_ACTIVE_MAX
unset IPT_DAVIDSON_CORRECTION_MAX_PER_STEP
unset IPT_DAVIDSON_DEBUG_ACTIVE_MAX

echo "===== NH3 k=30 automatic Ritz-cluster Davidson ====="
echo "TAG=$TAG"
echo "ROOT=$ROOT"
echo "partition=${SLURM_JOB_PARTITION:-gpu}"
echo "RESULT_DIR=$IPT_C_RESULTS_DIR"
echo "LOG_DIR=$IPT_C_LOG_DIR"
echo "manual_force_active_pairs_set=0"
echo "manual_active_clusters_set=0"
echo "manual_active_tol_set=0"
echo "manual_active_max_set=0"
echo "manual_correction_max_per_step_set=0"

exec bash "$TEST_DIR/run_nh3_k30_ipt_auto_cluster.sh"