#!/usr/bin/env bash
#SBATCH --job-name=ch4_block_dav
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/block_active_davidson_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/block_active_davidson_%j.err

set -u -o pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
BASE="$ROOT/results/sparse/CH4_K30_debug/block_active_davidson_v1"
LOG_BASE="$ROOT/logs/sparse/CH4_K30_dubug/block_active_davidson_v1"
CACHE="$ROOT/results/sparse/CH4_K30_debug/ch4_sto6g_fci_15876_csc.bin"
RUNNER="$ROOT/test/sparse/CH4/run_ch4_sto6g_fci_15876_ipt.sh"
OUTPUT="$BASE/block_active_davidson_sweep.csv"
BEST="$BASE/block_active_davidson_best.txt"
build_needed=1

mkdir -p "$BASE" "$LOG_BASE"
printf '%s\n' \
    "denom_clip,extra_steps,ortho_repeats,exit_code,status,baseline_steps,baseline_max_relative_eigen_residual,baseline_pair_28_residual,baseline_pair_29_residual,baseline_basis_cols,baseline_orthogonality_error,accepted_rounds,accepted_corrections,rejected_corrections,final_max_relative_eigen_residual,final_max_relative_eigen_residual_index,final_pair_28_residual,final_pair_29_residual,final_basis_cols,final_orthogonality_error,pair_28_linear_dependent_count,pair_29_linear_dependent_count,result_dir" \
    >"$OUTPUT"

for denom in 3e-9 1e-8; do
    for extra_steps in 10 20 40; do
        for ortho in 2 3; do
            case_name="denom_${denom}_extra_${extra_steps}_ortho_${ortho}"
            result_dir="$BASE/$case_name"
            log_dir="$LOG_BASE/$case_name"
            trial_csv="$result_dir/ch4_sto6g_fci_15876_ipt_relative_trials.csv"
            pair_csv="$result_dir/ch4_sto6g_fci_15876_pair_residuals.csv"
            block_csv="$result_dir/ch4_sto6g_fci_15876_davidson_block_history.csv"
            selection_csv="$result_dir/ch4_sto6g_fci_15876_davidson_selection_history.csv"

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
                export IPT_DAVIDSON_STEPS=30
                export IPT_DAVIDSON_EXTRA_STEPS="$extra_steps"
                export IPT_DAVIDSON_ACTIVE_MAX=2
                export IPT_DAVIDSON_PROTECT_TOL=1e-10
                export IPT_DAVIDSON_CONVERGED_TOL=1e-13
                export IPT_DAVIDSON_ACTIVE_TOL=1e-13
                export IPT_DAVIDSON_FORCE_ACTIVE_PAIRS=28,29
                export IPT_DAVIDSON_BLOCK_ACTIVE=1
                export IPT_DAVIDSON_DENOM_CLIP="$denom"
                export IPT_DAVIDSON_ACCEPT_ONLY_IF_IMPROVES=1
                export IPT_DAVIDSON_ORTHO_REPEATS="$ortho"
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

            if [[ -s "$trial_csv" && -s "$pair_csv" &&
                  -s "$block_csv" && -s "$selection_csv" ]]; then
                status="$(awk -F, 'NR == 2 {print $7}' "$trial_csv")"
                final_basis="$(awk -F, 'NR == 2 {print $4}' "$trial_csv")"
                final_max="$(awk -F, 'NR == 2 {print $9}' "$trial_csv")"
                final_index="$(awk -F, 'NR == 2 {print $10}' "$trial_csv")"
                final_28="$(awk -F, 'NR > 1 && $6 == 28 {print $8}' "$pair_csv")"
                final_29="$(awk -F, 'NR > 1 && $6 == 29 {print $8}' "$pair_csv")"
                baseline_max="$(awk -F, 'NR == 2 {print $7}' "$block_csv")"
                baseline_28="$(awk -F, 'NR == 2 {print $9}' "$block_csv")"
                baseline_29="$(awk -F, 'NR == 2 {print $11}' "$block_csv")"
                baseline_basis="$(awk -F, 'NR == 2 {print $15}' "$block_csv")"
                baseline_orth="$(awk -F, 'NR == 2 {print $17}' "$block_csv")"
                accepted_rounds="$(awk -F, 'NR > 1 && $13 == 1 {count++} END {print count + 0}' "$block_csv")"
                accepted_corrections="$(awk -F, 'NR > 1 {count += $3} END {print count + 0}' "$block_csv")"
                rejected_corrections="$(awk -F, 'NR > 1 {count += $4} END {print count + 0}' "$block_csv")"
                final_orth="$(awk -F, 'NR > 1 {value=$18} END {print value}' "$block_csv")"
                pair_28_dependent="$(awk -F, 'NR > 1 && $1 > 30 && $2 == 28 && $8 == 1 {count++} END {print count + 0}' "$selection_csv")"
                pair_29_dependent="$(awk -F, 'NR > 1 && $1 > 30 && $2 == 29 && $8 == 1 {count++} END {print count + 0}' "$selection_csv")"
                printf '%s\n' \
                    "$denom,$extra_steps,$ortho,$exit_code,$status,30,$baseline_max,$baseline_28,$baseline_29,$baseline_basis,$baseline_orth,$accepted_rounds,$accepted_corrections,$rejected_corrections,$final_max,$final_index,$final_28,$final_29,$final_basis,$final_orth,$pair_28_dependent,$pair_29_dependent,$result_dir" \
                    >>"$OUTPUT"
            else
                printf '%s\n' \
                    "$denom,$extra_steps,$ortho,$exit_code,missing,30,nan,nan,nan,0,nan,0,0,0,nan,-1,nan,nan,0,nan,0,0,$result_dir" \
                    >>"$OUTPUT"
            fi
        done
    done
done

awk -F, '
    NR > 1 && $15 != "nan" && (best == "" || $15 + 0 < best + 0) {
        best = $15
        line = $0
    }
    END {
        print "selection_metric=final_max_relative_eigen_residual"
        print "best_row=" line
    }
' "$OUTPUT" >"$BEST"

cat "$OUTPUT"
cat "$BEST"
