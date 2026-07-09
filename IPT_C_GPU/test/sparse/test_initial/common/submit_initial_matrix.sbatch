#!/usr/bin/env bash
#SBATCH --job-name=tim_case
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/test_initial/matrix_%x_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/test_initial/matrix_%x_%j.err

set -uo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
COMMON_DIR="$ROOT/test/sparse/test_initial/common"
CASE_ENV="${1:?usage: submit_initial_matrix.sbatch CASE_ENV}"
K_LIST="${K_LIST:-10 20 30}"

# shellcheck disable=SC1090
source "$CASE_ENV"
: "${IPT_MATRIX_ID:?IPT_MATRIX_ID is required by case env}"

echo "===== test_initial matrix $IPT_MATRIX_ID ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CASE_ENV=$CASE_ENV"
echo "K_LIST=$K_LIST"
echo "IPT_MAX_LIFT_BLOCK_SIZE=${IPT_MAX_LIFT_BLOCK_SIZE:-default_64}"

overall_status=0
for k in $K_LIST; do
    summary="$ROOT/results/sparse/test_initial/$IPT_MATRIX_ID/K$k/${IPT_MATRIX_ID}_summary.txt"
    if [ -f "$summary" ] && grep -q 'method=IPT_CUDA_BLOCK_CLUSTER .*status=success' "$summary"; then
        echo "SKIP_SUCCESS $IPT_MATRIX_ID K=$k summary=$summary"
        continue
    fi

    echo "----- RUN $IPT_MATRIX_ID K=$k -----"
    if bash "$COMMON_DIR/run_initial_case.sh" "$CASE_ENV" "$k"; then
        echo "OK $IPT_MATRIX_ID K=$k"
    else
        status=$?
        overall_status=$status
        echo "FAILED $IPT_MATRIX_ID K=$k status=$status"
    fi
done

echo "finished matrix $IPT_MATRIX_ID at: $(date)"
exit "$overall_status"
