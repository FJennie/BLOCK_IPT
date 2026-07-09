#!/usr/bin/env bash
set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
COMMON_DIR="$ROOT/test/sparse/test_initial/common"
BIN="${IPT_BENCH_BIN:-$COMMON_DIR/.build/benchmark_initial_campaign_sm70}"
K_LIST="${K_LIST:-10 20 30}"
PARTITION="${PARTITION:-v100}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MAX_LIFT_BLOCK_SIZE="${IPT_MAX_LIFT_BLOCK_SIZE:-128}"
SUBMIT_LOG_DIR="$ROOT/results/sparse/test_initial"
SUBMIT_LOG="$SUBMIT_LOG_DIR/submitted_jobs_$(date +%Y%m%d_%H%M%S).tsv"

mkdir -p "$SUBMIT_LOG_DIR"
printf "job_id\tmatrix_id\tk_list\tcase_env\tpartition\tmax_lift_block_size\n" > "$SUBMIT_LOG"

for case_env in "$ROOT"/test/sparse/test_initial/*/case.env; do
    unset IPT_MATRIX_ID IPT_MOLECULE_ATOMS IPT_MOLECULE_BASIS
    unset IPT_ACTIVE_ELECTRONS IPT_ACTIVE_ORBITALS IPT_CHARGE IPT_SPIN
    # shellcheck disable=SC1090
    source "$case_env"

    all_success=1
    for k in $K_LIST; do
        summary="$ROOT/results/sparse/test_initial/$IPT_MATRIX_ID/K$k/${IPT_MATRIX_ID}_summary.txt"
        if ! { [ -f "$summary" ] && grep -q 'method=IPT_CUDA_BLOCK_CLUSTER .*status=success' "$summary"; }; then
            all_success=0
            break
        fi
    done

    if [ "$all_success" = "1" ]; then
        printf "SKIP_SUCCESS\t%s\t%s\t%s\t%s\t%s\n" \
            "$IPT_MATRIX_ID" "$K_LIST" "$case_env" "$PARTITION" \
            "$MAX_LIFT_BLOCK_SIZE" | tee -a "$SUBMIT_LOG"
        continue
    fi

    job=$(
        sbatch \
            --job-name="tim_${IPT_MATRIX_ID}" \
            --partition="$PARTITION" \
            --time="$TIME_LIMIT" \
            --export=ALL,IPT_SKIP_BUILD=1,IPT_BENCH_BIN="$BIN",IPT_MAX_LIFT_BLOCK_SIZE="$MAX_LIFT_BLOCK_SIZE" \
            "$COMMON_DIR/submit_initial_matrix.sbatch" \
            "$case_env" | awk '{print $4}'
    )
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$job" "$IPT_MATRIX_ID" "$K_LIST" "$case_env" "$PARTITION" \
        "$MAX_LIFT_BLOCK_SIZE" | tee -a "$SUBMIT_LOG"
done

echo "submitted_jobs=$SUBMIT_LOG"
