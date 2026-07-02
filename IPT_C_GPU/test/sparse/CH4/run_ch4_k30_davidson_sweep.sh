#!/usr/bin/env bash
#SBATCH --job-name=ch4_davidson
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/davidson_sweep_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/davidson_sweep_%j.err

set -u -o pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
RESULTS_BASE="${IPT_DAVIDSON_RESULTS_BASE:-$ROOT/results/sparse/CH4_K30_debug/davidson_sweep}"
LOGS_BASE="${IPT_DAVIDSON_LOGS_BASE:-$ROOT/logs/sparse/CH4_K30_dubug/davidson_sweep}"
CACHE="${IPT_CH4_MATRIX_CACHE:-$ROOT/results/sparse/CH4_K30_debug/ch4_sto6g_fci_15876_csc.bin}"
STEPS_LIST="${IPT_DAVIDSON_STEPS_LIST:-1 2 3 5 10 20}"
CLIP_LIST="${IPT_DAVIDSON_CLIP_LIST:-1e-6 1e-8 1e-10 1e-12}"
ACTIVE_LIST="${IPT_DAVIDSON_ACTIVE_LIST:-1 2}"
RUNNER="$ROOT/test/sparse/CH4/run_ch4_sto6g_fci_15876_ipt.sh"
SWEEP_CSV="$RESULTS_BASE/davidson_sweep.csv"
BEST_TXT="$RESULTS_BASE/davidson_best.txt"

mkdir -p "$RESULTS_BASE" "$LOGS_BASE"
printf '%s\n' \
    "steps,denom_clip,active_max,exit_code,status,basis_cols,accepted_entries,max_relative_eigen_residual,max_relative_eigen_residual_index,pair_28_residual,pair_29_residual,max_pair_0_28_residual,result_dir" \
    >"$SWEEP_CSV"

for steps in $STEPS_LIST; do
    for denom_clip in $CLIP_LIST; do
        for active_max in $ACTIVE_LIST; do
            case_name="steps_${steps}_clip_${denom_clip}_active_${active_max}"
            result_dir="$RESULTS_BASE/$case_name"
            log_dir="$LOGS_BASE/$case_name"
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
                export IPT_DAVIDSON_ACTIVE_MAX="$active_max"
                export IPT_DAVIDSON_SELECT_TOL=1e-10
                export IPT_DAVIDSON_DENOM_CLIP="$denom_clip"
                export IPT_DAVIDSON_ACCEPT_ONLY_IF_IMPROVES=1
                export IPT_DAVIDSON_ORTHO_REPEATS=2
                export IPT_SKIP_BUILD=1
                export RUN_PRIMME=0
                export RUN_WARMUP=0
                bash "$RUNNER"
            )
            exit_code=$?

            if [[ -s "$trial_csv" && -s "$pair_csv" ]]; then
                status="$(awk -F, 'NR == 2 {print $7}' "$trial_csv")"
                basis_cols="$(awk -F, 'NR == 2 {print $4}' "$trial_csv")"
                maximum="$(awk -F, 'NR == 2 {print $9}' "$trial_csv")"
                maximum_index="$(awk -F, 'NR == 2 {print $10}' "$trial_csv")"
                pair_28="$(awk -F, 'NR > 1 && $6 == 28 {print $8}' "$pair_csv")"
                pair_29="$(awk -F, 'NR > 1 && $6 == 29 {print $8}' "$pair_csv")"
                max_0_28="$(awk -F, '
                    NR > 1 && $6 < 29 && (maximum == "" || $8 + 0 > maximum + 0) {
                        maximum = $8
                    }
                    END {print maximum}
                ' "$pair_csv")"
                accepted_entries="$(awk -F, '
                    NR > 1 && $5 == 1 {count++}
                    END {print count + 0}
                ' "$history_csv")"
                printf '%s\n' \
                    "$steps,$denom_clip,$active_max,$exit_code,$status,$basis_cols,$accepted_entries,$maximum,$maximum_index,$pair_28,$pair_29,$max_0_28,$result_dir" \
                    >>"$SWEEP_CSV"
            else
                printf '%s\n' \
                    "$steps,$denom_clip,$active_max,$exit_code,missing,0,0,nan,-1,nan,nan,nan,$result_dir" \
                    >>"$SWEEP_CSV"
            fi
        done
    done
done

awk -F, '
    NR > 1 && $8 != "nan" && (best == "" || $8 + 0 < best + 0) {
        best = $8
        line = $0
    }
    END {
        print "selection_metric=max_relative_eigen_residual"
        print "best_row=" line
    }
' "$SWEEP_CSV" >"$BEST_TXT"

cat "$SWEEP_CSV"
cat "$BEST_TXT"
