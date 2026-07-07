#!/usr/bin/env bash
#SBATCH --job-name=ch4_block_base
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/block_active_baseline_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/block_active_baseline_%j.err

set -u -o pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
BASE="$ROOT/results/sparse/CH4_K30_debug/block_active_davidson_v1/baseline_compare"
LOG_BASE="$ROOT/logs/sparse/CH4_K30_dubug/block_active_davidson_v1/baseline_compare"
CACHE="$ROOT/results/sparse/CH4_K30_debug/ch4_sto6g_fci_15876_csc.bin"
RUNNER="$ROOT/test/sparse/CH4/run_ch4_sto6g_fci_15876_ipt.sh"
OUTPUT="$BASE/baseline_compare.csv"
build_needed=1

mkdir -p "$BASE" "$LOG_BASE"
printf '%s\n' \
    "baseline_steps,denom_clip,exit_code,status,baseline_max_relative_eigen_residual,pair_28_residual,pair_29_residual,basis_cols,accepted_entries,orthogonality_error,result_dir" \
    >"$OUTPUT"

for steps in 30 40; do
    for denom in 3e-9 1e-8; do
        case_name="steps_${steps}_denom_${denom}"
        result_dir="$BASE/$case_name"
        log_dir="$LOG_BASE/$case_name"
        trial_csv="$result_dir/ch4_sto6g_fci_15876_ipt_relative_trials.csv"
        pair_csv="$result_dir/ch4_sto6g_fci_15876_pair_residuals.csv"
        history_csv="$result_dir/ch4_sto6g_fci_15876_davidson_history.csv"

        mkdir -p "$result_dir" "$log_dir"
        (
            export IPT_C_RESULTS_DIR="$result_dir"
            export IPT_C_LOG_DIR="$log_dir"
            export IPT_CH4_MATRIX_CACHE="$CACHE"
            export IPT_LOAD_MATRIX=1
            export IPT_SAVE_MATRIX=0
            export IPT_K=30
            export IPT_REPEATS=1
            export IPT_MAXITER=200
            export IPT_BLOCK_CLUSTER_MAXITER=200
            export IPT_BLOCK_CLUSTER_TOL=0
            export IPT_BLOCK_CLUSTER=1
            export IPT_BLOCK_CLUSTER_OVERSAMPLE=3
            export IPT_BLOCK_CLUSTER_QR=1
            export IPT_BLOCK_CLUSTER_ADAPTIVE=0
            export IPT_DAVIDSON_ENRICH=1
            export IPT_DAVIDSON_STEPS="$steps"
            export IPT_DAVIDSON_EXTRA_STEPS=0
            export IPT_DAVIDSON_ACTIVE_MAX=2
            export IPT_DAVIDSON_PROTECT_TOL=1e-10
            export IPT_DAVIDSON_CONVERGED_TOL=1e-13
            export IPT_DAVIDSON_FORCE_ACTIVE_PAIRS=
            export IPT_DAVIDSON_BLOCK_ACTIVE=0
            export IPT_DAVIDSON_DENOM_CLIP="$denom"
            export IPT_DAVIDSON_ACCEPT_ONLY_IF_IMPROVES=1
            export IPT_DAVIDSON_ORTHO_REPEATS=2
            export IPT_DAVIDSON_RESTART_EVERY=20
            export IPT_DAVIDSON_RESTART_KEEP_EXTRA=5
            export IPT_JD_LOCAL_CORRECTION=0
            export RUN_PRIMME=0
            export RUN_WARMUP=0
            export IPT_SKIP_BUILD="$((1 - build_needed))"
            bash "$RUNNER"
        )
        exit_code=$?
        build_needed=0

        if [[ -s "$trial_csv" && -s "$pair_csv" ]]; then
            status="$(awk -F, 'NR == 2 {print $7}' "$trial_csv")"
            basis_cols="$(awk -F, 'NR == 2 {print $4}' "$trial_csv")"
            maximum="$(awk -F, 'NR == 2 {print $9}' "$trial_csv")"
            pair_28="$(awk -F, 'NR > 1 && $6 == 28 {print $8}' "$pair_csv")"
            pair_29="$(awk -F, 'NR > 1 && $6 == 29 {print $8}' "$pair_csv")"
            accepted_entries="$(awk -F, 'NR > 1 && $5 == 1 {count++} END {print count + 0}' "$history_csv")"
            orthogonality="$(awk -F, 'NR > 1 {value=$11} END {print value}' "$history_csv")"
            printf '%s\n' \
                "$steps,$denom,$exit_code,$status,$maximum,$pair_28,$pair_29,$basis_cols,$accepted_entries,$orthogonality,$result_dir" \
                >>"$OUTPUT"
        fi
    done
done

cat "$OUTPUT"
