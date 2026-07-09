#!/usr/bin/env bash
set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
COMMON_DIR="$ROOT/test/sparse/test_initial/common"
RESULT_BASE="${IPT_RESULT_BASE:-$ROOT/results/sparse/test_initial}"
PARAM_SET="${PARAM_SET:-default}"
K_LIST="${K_LIST:-10 20 30}"
PARTITION="${PARTITION:-v100}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
MAX_SUBMIT="${MAX_SUBMIT:-0}"
SUBMIT_LOG="$RESULT_BASE/submitted_cache_${PARAM_SET}_$(date +%Y%m%d_%H%M%S).tsv"

mkdir -p "$RESULT_BASE" "$ROOT/logs/sparse/test_initial"
printf "job_id\tmatrix_id\tk_list\tparam_set\tpartition\tmatrix_cache\n" > "$SUBMIT_LOG"

queued_jobs="$(squeue -h -u "${USER:-nudt_liujie}" -o "%j" 2>/dev/null || true)"
submitted=0

while IFS= read -r cache; do
    [ -n "$cache" ] || continue
    matrix_dir="$(dirname "$cache")"
    matrix_id="$(basename "$matrix_dir")"
    job_name="tic_${matrix_id}_${PARAM_SET}"
    job_name="${job_name:0:120}"

    if printf "%s\n" "$queued_jobs" | grep -Fxq "$job_name"; then
        echo "skip queued $matrix_id"
        continue
    fi

    complete=1
    for k in $K_LIST; do
        status="$RESULT_BASE/$matrix_id/K$k/$PARAM_SET/run_status.tsv"
        if [ ! -f "$status" ]; then
            complete=0
            break
        fi
    done
    if [ "$complete" = "1" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "skip complete $matrix_id"
        continue
    fi

    out=$(
        sbatch \
            --job-name="$job_name" \
            --partition="$PARTITION" \
            --time="$TIME_LIMIT" \
            --export=ALL,PARAM_SET="$PARAM_SET",K_LIST="$K_LIST" \
            "$COMMON_DIR/submit_initial_cache_matrix.sbatch" \
            "$matrix_dir"
    )
    job="$(printf "%s\n" "$out" | awk '{print $4}')"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$job" "$matrix_id" "$K_LIST" "$PARAM_SET" "$PARTITION" "$cache" | tee -a "$SUBMIT_LOG"
    submitted=$((submitted + 1))
    if [ "$MAX_SUBMIT" -gt 0 ] && [ "$submitted" -ge "$MAX_SUBMIT" ]; then
        break
    fi
done < <(find "$ROOT/test/sparse/test_initial" -mindepth 2 -maxdepth 2 -type f -name '*_csc.bin' | sort)

echo "submitted=$submitted"
echo "submitted_jobs=$SUBMIT_LOG"
