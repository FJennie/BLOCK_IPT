#!/usr/bin/env bash
#SBATCH --job-name=tic_matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/test_initial/slurm_%x_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/test_initial/slurm_%x_%j.err

set -uo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
COMMON_DIR="$ROOT/test/sparse/test_initial/common"
MATRIX_REF="${1:?usage: submit_initial_cache_matrix.sbatch MATRIX_DIR_OR_ID}"
K_LIST="${K_LIST:-10 20 30}"
PARAM_SET="${PARAM_SET:-default}"

echo "===== cache-only matrix job ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "MATRIX_REF=$MATRIX_REF"
echo "K_LIST=$K_LIST"
echo "PARAM_SET=$PARAM_SET"

for k in $K_LIST; do
    echo "----- RUN $MATRIX_REF K=$k PARAM=$PARAM_SET -----"
    bash "$COMMON_DIR/run_initial_cache_case.sh" "$MATRIX_REF" "$k" "$PARAM_SET"
done

echo "finished matrix job at: $(date)"
exit 0
