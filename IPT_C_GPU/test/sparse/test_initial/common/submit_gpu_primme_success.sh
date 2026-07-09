#!/usr/bin/env bash
set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
COMMON_DIR="$ROOT/test/sparse/test_initial/common"
LIST="${GPU_PRIMME_LIST:-$ROOT/results/sparse/test_initial_gpu_primme/gpu_primme_success_k_list.tsv}"
BIN="${IPT_BENCH_BIN:-$COMMON_DIR/.build/benchmark_initial_campaign_sm70_gpuprimme}"
PARTITION="${PARTITION:-v100}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
RESULT_BASE="${IPT_RESULT_BASE:-$ROOT/results/sparse/test_initial_gpu_primme}"
LOG_BASE="${IPT_LOG_BASE:-$ROOT/logs/sparse/test_initial_gpu_primme}"
SUBMIT_LOG="$RESULT_BASE/submitted_gpu_primme_$(date +%Y%m%d_%H%M%S).tsv"

mkdir -p "$RESULT_BASE" "$LOG_BASE"
printf "job_id\tmatrix_id\tk\tcase_env\tpartition\n" > "$SUBMIT_LOG"

queued_jobs="$(squeue -h -u "${USER:-nudt_liujie}" -o "%j" 2>/dev/null || true)"

tail -n +2 "$LIST" | while IFS=$'\t' read -r matrix_id k case_env _rest; do
    [ -n "$matrix_id" ] || continue
    summary="$RESULT_BASE/$matrix_id/K$k/${matrix_id}_summary.txt"
    if [ -f "$summary" ] && grep -q 'method=PRIMME_CUBLAS .*status=success' "$summary"; then
        echo "skip_success $matrix_id K$k"
        continue
    fi
    job_name="pg_${matrix_id}_${k}"
    if printf "%s\n" "$queued_jobs" | grep -Fxq "$job_name"; then
        echo "skip_queued $matrix_id K$k"
        continue
    fi

    set +e
    out=$(
        sbatch \
            --job-name="$job_name" \
            --partition="$PARTITION" \
            --time="$TIME_LIMIT" \
            --export=ALL,IPT_SKIP_BUILD=1,IPT_BENCH_BIN="$BIN",RUN_IPT=0,RUN_PRIMME=1,IPT_RESULT_BASE="$RESULT_BASE",IPT_LOG_BASE="$LOG_BASE" \
            "$COMMON_DIR/submit_initial_case.sbatch" \
            "$case_env" "$k" 2>&1
    )
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "submit_failed $matrix_id K$k $out"
        break
    fi
    job="$(printf "%s\n" "$out" | awk '{print $4}')"
    printf "%s\t%s\t%s\t%s\t%s\n" "$job" "$matrix_id" "$k" "$case_env" "$PARTITION" | tee -a "$SUBMIT_LOG"
done

echo "submitted_jobs=$SUBMIT_LOG"
