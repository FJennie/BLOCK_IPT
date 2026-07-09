#!/usr/bin/env bash
set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
COMMON_DIR="$ROOT/test/sparse/test_initial/common"
BIN="${IPT_BENCH_BIN:-$COMMON_DIR/.build/benchmark_initial_campaign_sm70}"
CASE_LIST="${CASE_LIST:-$COMMON_DIR/diverse_matrix_order.txt}"
K_LIST="${K_LIST:-10 20 30}"
PARTITION="${PARTITION:-v100}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MAX_LIFT_BLOCK_SIZE="${IPT_MAX_LIFT_BLOCK_SIZE:-128}"
MAX_TESTED_MATRICES="${MAX_TESTED_MATRICES:-100}"
TARGET_ALLK_SUCCESS="${TARGET_ALLK_SUCCESS:-20}"
SUBMIT_LOG_DIR="$ROOT/results/sparse/test_initial"
SUBMIT_LOG="$SUBMIT_LOG_DIR/submitted_jobs_selected_$(date +%Y%m%d_%H%M%S).tsv"

mkdir -p "$SUBMIT_LOG_DIR"
printf "job_id\tmatrix_id\tk_list\tcase_env\tpartition\tmax_lift_block_size\n" > "$SUBMIT_LOG"

python3 "$COMMON_DIR/collect_initial_results.py" >/dev/null || true
read -r tested_count allk_success_count < <(
    python3 - <<'PY'
import csv
from collections import defaultdict

p = "/fs1/home/nudt_liujie/ftt/IPT_C_GPU/results/sparse/test_initial/initial_results_summary.csv"
rows = list(csv.DictReader(open(p))) if __import__("os").path.exists(p) else []
by = defaultdict(list)
tested = set()
for r in rows:
    by[r["matrix_id"]].append(r)
    if r.get("tested") == "1":
        tested.add(r["matrix_id"])
allk = 0
for matrix_id, rs in by.items():
    if all(any(x["k"] == k and x["ipt_status"] == "success" for x in rs) for k in ("10", "20", "30")):
        allk += 1
print(len(tested), allk)
PY
)

echo "tested_matrices=$tested_count allK_success_matrices=$allk_success_count"
if [ "$allk_success_count" -ge "$TARGET_ALLK_SUCCESS" ]; then
    echo "target reached; no submission needed"
    exit 0
fi
if [ "$tested_count" -ge "$MAX_TESTED_MATRICES" ]; then
    echo "tested matrix limit reached; no submission needed"
    exit 0
fi

queued_jobs="$(squeue -h -u "${USER:-nudt_liujie}" -o "%j" 2>/dev/null || true)"
submitted=0
while IFS= read -r matrix_id; do
    matrix_id="${matrix_id%%#*}"
    matrix_id="$(printf "%s" "$matrix_id" | tr -d '[:space:]')"
    [ -n "$matrix_id" ] || continue

    case_env="$ROOT/test/sparse/test_initial/$matrix_id/case.env"
    if [ ! -f "$case_env" ]; then
        echo "skip missing case: $matrix_id"
        continue
    fi

    if printf "%s\n" "$queued_jobs" | grep -Fxq "tim_$matrix_id"; then
        echo "skip queued: $matrix_id"
        continue
    fi

    all_success=1
    complete_attempt=1
    for k in $K_LIST; do
        summary="$ROOT/results/sparse/test_initial/$matrix_id/K$k/${matrix_id}_summary.txt"
        log="$ROOT/logs/sparse/test_initial/$matrix_id/K$k/latest.log"
        if ! { [ -f "$summary" ] && grep -q 'method=IPT_CUDA_BLOCK_CLUSTER .*status=success' "$summary"; }; then
            all_success=0
        fi
        if ! { [ -f "$summary" ] || [ -f "$log" ]; }; then
            complete_attempt=0
        fi
    done

    if [ "$all_success" = "1" ]; then
        echo "skip allK success: $matrix_id"
        continue
    fi
    if [ "$complete_attempt" = "1" ]; then
        echo "skip already attempted: $matrix_id"
        continue
    fi

    if [ "$tested_count" -ge "$MAX_TESTED_MATRICES" ]; then
        echo "tested matrix limit reached during submission"
        break
    fi

    set +e
    out=$(
        sbatch \
            --job-name="tim_${matrix_id}" \
            --partition="$PARTITION" \
            --time="$TIME_LIMIT" \
            --export=ALL,IPT_SKIP_BUILD=1,IPT_BENCH_BIN="$BIN",IPT_MAX_LIFT_BLOCK_SIZE="$MAX_LIFT_BLOCK_SIZE" \
            "$COMMON_DIR/submit_initial_matrix.sbatch" \
            "$case_env" 2>&1
    )
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "submit_failed $matrix_id $out"
        break
    fi
    job="$(printf "%s\n" "$out" | awk '{print $4}')"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$job" "$matrix_id" "$K_LIST" "$case_env" "$PARTITION" \
        "$MAX_LIFT_BLOCK_SIZE" | tee -a "$SUBMIT_LOG"
    submitted=$((submitted + 1))
    tested_count=$((tested_count + 1))
done < "$CASE_LIST"

echo "submitted=$submitted"
echo "submitted_jobs=$SUBMIT_LOG"
