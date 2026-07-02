#!/usr/bin/env bash
#SBATCH --job-name=ch4_k30_sweep
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/checkpoint_sweep_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/checkpoint_sweep_%j.err

set -u -o pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
RESULTS_BASE="${IPT_CHECKPOINT_RESULTS_BASE:-$ROOT/results/sparse/CH4_K30_debug/checkpoint_sweep}"
LOGS_BASE="${IPT_CHECKPOINT_LOGS_BASE:-$ROOT/logs/sparse/CH4_K30_dubug/checkpoint_sweep}"
CACHE="${IPT_CH4_MATRIX_CACHE:-$ROOT/results/sparse/CH4_K30_debug/ch4_sto6g_fci_15876_csc.bin}"
ITERATIONS="${IPT_CHECKPOINT_ITERS:-50 100 150 200 300 400 500 700 1000}"
RUNNER="$ROOT/test/sparse/CH4/run_ch4_sto6g_fci_15876_ipt.sh"
SWEEP_CSV="$RESULTS_BASE/checkpoint_sweep.csv"
BEST_TXT="$RESULTS_BASE/checkpoint_best.txt"

mkdir -p "$RESULTS_BASE" "$LOGS_BASE"
printf '%s\n' \
    "iterations,exit_code,status,basis_cols,returned_k,max_relative_eigen_residual,max_relative_eigen_residual_index,result_dir" \
    >"$SWEEP_CSV"

for iteration in $ITERATIONS; do
    result_dir="$RESULTS_BASE/iter_$iteration"
    log_dir="$LOGS_BASE/iter_$iteration"
    trial_csv="$result_dir/ch4_sto6g_fci_15876_ipt_relative_trials.csv"

    mkdir -p "$result_dir" "$log_dir"
    (
        export IPT_C_RESULTS_DIR="$result_dir"
        export IPT_C_LOG_DIR="$log_dir"
        export IPT_CH4_MATRIX_CACHE="$CACHE"
        export IPT_LOAD_MATRIX=1
        export IPT_SAVE_MATRIX=0
        export IPT_K=30
        export IPT_REPEATS=1
        export IPT_MAXITER="$iteration"
        export IPT_BLOCK_CLUSTER_MAXITER="$iteration"
        export IPT_BLOCK_CLUSTER_TOL=0
        export IPT_BLOCK_CLUSTER=1
        export IPT_BLOCK_CLUSTER_OVERSAMPLE=3
        export IPT_BLOCK_CLUSTER_QR=1
        export IPT_BLOCK_CLUSTER_ADAPTIVE=0
        export IPT_DAVIDSON_ENRICH=0
        export IPT_SKIP_BUILD=1
        export RUN_PRIMME=0
        export RUN_WARMUP=0
        bash "$RUNNER"
    )
    exit_code=$?

    if [[ -s "$trial_csv" ]]; then
        awk -F, -v iter="$iteration" -v rc="$exit_code" \
            -v dir="$result_dir" \
            'NR == 2 {
                printf "%d,%d,%s,%s,%s,%s,%s,%s\n",
                       iter, rc, $7, $4, $5, $9, $10, dir
            }' "$trial_csv" >>"$SWEEP_CSV"
    else
        printf '%s\n' \
            "$iteration,$exit_code,missing,0,0,nan,-1,$result_dir" \
            >>"$SWEEP_CSV"
    fi
done

awk -F, '
    NR > 1 && $6 != "nan" && (best == "" || $6 + 0 < best + 0) {
        best = $6
        line = $0
    }
    END {
        print "selection_metric=max_relative_eigen_residual"
        print "best_row=" line
    }
' "$SWEEP_CSV" >"$BEST_TXT"

cat "$SWEEP_CSV"
cat "$BEST_TXT"
