#!/usr/bin/env bash
#SBATCH --job-name=h2o_k30_auto
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/H2O_k30/h2o_k30_auto_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/H2O_k30/h2o_k30_auto_%j.err

set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
TEST_DIR="$ROOT/test/sparse/H2O"
STAMP="$(date +%Y%m%d_%H%M%S)"
TAG="${COMPARE_TAG:-h2o_k30_auto_${STAMP}}"

export IPT_C_RESULTS_DIR="$ROOT/results/sparse/H2O_k30/$TAG"
export IPT_C_LOG_DIR="$ROOT/logs/sparse/H2O_k30/$TAG"
export IPT_H2O_MATRIX_CACHE="${IPT_H2O_MATRIX_CACHE:-$TEST_DIR/h2o_sto6g_fci_441_csc.bin}"
export IPT_LOAD_MATRIX="${IPT_LOAD_MATRIX:-1}"
export IPT_SAVE_MATRIX="${IPT_SAVE_MATRIX:-1}"

mkdir -p "$IPT_C_RESULTS_DIR" "$IPT_C_LOG_DIR" "$ROOT/logs/sparse/H2O_k30"

unset IPT_COMPARE_FORCE_ACTIVE_PAIRS
unset IPT_DAVIDSON_FORCE_ACTIVE_PAIRS
unset IPT_DAVIDSON_ACTIVE_CLUSTERS
unset IPT_DAVIDSON_ACTIVE_TOL
unset IPT_DAVIDSON_ACTIVE_MAX
unset IPT_DAVIDSON_CORRECTION_MAX_PER_STEP
unset IPT_DAVIDSON_DEBUG_ACTIVE_MAX

echo "===== H2O k=30 automatic Ritz-cluster IPT vs PRIMME ====="
echo "TAG=$TAG"
echo "ROOT=$ROOT"
echo "partition=${SLURM_JOB_PARTITION:-gpu}"
echo "RESULT_DIR=$IPT_C_RESULTS_DIR"
echo "LOG_DIR=$IPT_C_LOG_DIR"
echo "MATRIX_CACHE=$IPT_H2O_MATRIX_CACHE"
echo "manual_force_active_pairs_set=0"
echo "manual_active_clusters_set=0"
echo "manual_active_tol_set=0"
echo "manual_active_max_set=0"
echo "manual_correction_max_per_step_set=0"

exec bash "$TEST_DIR/run_h2o_k30_ipt_auto_cluster.sh"
