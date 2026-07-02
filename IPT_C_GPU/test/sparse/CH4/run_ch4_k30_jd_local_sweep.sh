#!/usr/bin/env bash
#SBATCH --job-name=ch4_jd_local
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/jd_local_sweep_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/jd_local_sweep_%j.err

set -u -o pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
RESULTS_BASE="${IPT_JD_RESULTS_BASE:-$ROOT/results/sparse/CH4_K30_debug/jd_local_sweep_v1}"
LOGS_BASE="${IPT_JD_LOGS_BASE:-$ROOT/logs/sparse/CH4_K30_dubug/jd_local_sweep_v1}"
CACHE="${IPT_CH4_MATRIX_CACHE:-$ROOT/results/sparse/CH4_K30_debug/ch4_sto6g_fci_15876_csc.bin}"
RUNNER="$ROOT/test/sparse/CH4/run_ch4_sto6g_fci_15876_ipt.sh"
ACTIVE_LIST="${IPT_JD_SWEEP_ACTIVE_LIST:-29 28,29}"
WINDOW_LIST="${IPT_JD_SWEEP_WINDOW_LIST:-25:40 20:45 1:32}"
DAMPING_LIST="${IPT_JD_SWEEP_DAMPING_LIST:-1e-6 1e-8 1e-10}"
OUTSIDE_LIST="${IPT_JD_SWEEP_OUTSIDE_LIST:-none diagonal}"
STEPS_LIST="${IPT_JD_SWEEP_STEPS_LIST:-1 3 5}"
SWEEP_CSV="$RESULTS_BASE/jd_local_sweep.csv"
BEST_TXT="$RESULTS_BASE/jd_local_best.txt"
build_needed=1

mkdir -p "$RESULTS_BASE" "$LOGS_BASE"
printf '%s\n' \
    "active_pairs,window_start,window_end,damping,outside_correction,steps,exit_code,status,basis_cols,accepted_entries,max_relative_eigen_residual,max_relative_eigen_residual_index,pair_28_residual,pair_29_residual,max_pair_0_27_residual,orthogonality_error,result_dir" \
    >"$SWEEP_CSV"

for active_pairs in $ACTIVE_LIST; do
    for window in $WINDOW_LIST; do
        window_start="${window%%:*}"
        window_end="${window##*:}"
        for damping in $DAMPING_LIST; do
            for outside in $OUTSIDE_LIST; do
                for steps in $STEPS_LIST; do
                    active_tag="${active_pairs//,/_}"
                    case_name="active_${active_tag}_window_${window_start}_${window_end}_damp_${damping}_outside_${outside}_steps_${steps}"
                    result_dir="$RESULTS_BASE/$case_name"
                    log_dir="$LOGS_BASE/$case_name"
                    trial_csv="$result_dir/ch4_sto6g_fci_15876_ipt_relative_trials.csv"
                    pair_csv="$result_dir/ch4_sto6g_fci_15876_pair_residuals.csv"
                    history_csv="$result_dir/ch4_sto6g_fci_15876_jd_local_history.csv"

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
                        export IPT_DAVIDSON_STEPS=80
                        export IPT_DAVIDSON_ACTIVE_MAX=2
                        export IPT_DAVIDSON_SELECT_TOL=1e-12
                        export IPT_DAVIDSON_PROTECT_TOL=1e-10
                        export IPT_DAVIDSON_LOCKED_TOL=1e-12
                        export IPT_DAVIDSON_DENOM_CLIP=3e-9
                        export IPT_DAVIDSON_ACCEPT_ONLY_IF_IMPROVES=1
                        export IPT_DAVIDSON_ORTHO_REPEATS=2
                        export IPT_DAVIDSON_RESTART_EVERY=20
                        export IPT_DAVIDSON_RESTART_KEEP_EXTRA=5
                        export IPT_DUMP_RESIDUAL_SUPPORT=0
                        export IPT_JD_LOCAL_CORRECTION=1
                        export IPT_JD_LOCAL_ACTIVE_PAIRS="$active_pairs"
                        export IPT_JD_LOCAL_WINDOW_START="$window_start"
                        export IPT_JD_LOCAL_WINDOW_END="$window_end"
                        export IPT_JD_LOCAL_DAMPING="$damping"
                        export IPT_JD_LOCAL_MAX_DIM=80
                        export IPT_JD_LOCAL_OUTSIDE_CORRECTION="$outside"
                        export IPT_JD_LOCAL_STEPS="$steps"
                        export IPT_JD_ACCEPT_ONLY_IF_IMPROVES=1
                        if [[ "$build_needed" == "1" ]]; then
                            export IPT_SKIP_BUILD=0
                        else
                            export IPT_SKIP_BUILD=1
                        fi
                        export RUN_PRIMME=0
                        export RUN_WARMUP=0
                        bash "$RUNNER"
                    )
                    exit_code=$?
                    build_needed=0

                    if [[ -s "$trial_csv" && -s "$pair_csv" &&
                          -s "$history_csv" ]]; then
                        status="$(awk -F, 'NR == 2 {print $7}' "$trial_csv")"
                        basis_cols="$(awk -F, 'NR == 2 {print $4}' "$trial_csv")"
                        maximum="$(awk -F, 'NR == 2 {print $9}' "$trial_csv")"
                        maximum_index="$(awk -F, 'NR == 2 {print $10}' "$trial_csv")"
                        pair_28="$(awk -F, 'NR > 1 && $6 == 28 {print $8}' "$pair_csv")"
                        pair_29="$(awk -F, 'NR > 1 && $6 == 29 {print $8}' "$pair_csv")"
                        max_0_27="$(awk -F, '
                            NR > 1 && $6 < 28 && (maximum == "" || $8 + 0 > maximum + 0) {
                                maximum = $8
                            }
                            END {print maximum}
                        ' "$pair_csv")"
                        accepted_entries="$(awk -F, '
                            NR > 1 && $5 == 1 {count++}
                            END {print count + 0}
                        ' "$history_csv")"
                        orthogonality="$(awk -F, 'NR > 1 {value = $11} END {print value}' "$history_csv")"
                        printf '%s\n' \
                            "$active_tag,$window_start,$window_end,$damping,$outside,$steps,$exit_code,$status,$basis_cols,$accepted_entries,$maximum,$maximum_index,$pair_28,$pair_29,$max_0_27,$orthogonality,$result_dir" \
                            >>"$SWEEP_CSV"
                    else
                        printf '%s\n' \
                            "$active_tag,$window_start,$window_end,$damping,$outside,$steps,$exit_code,missing,0,0,nan,-1,nan,nan,nan,nan,$result_dir" \
                            >>"$SWEEP_CSV"
                    fi
                done
            done
        done
    done
done

awk -F, '
    NR > 1 && $11 != "nan" && (best == "" || $11 + 0 < best + 0) {
        best = $11
        line = $0
    }
    END {
        print "selection_metric=max_relative_eigen_residual"
        print "best_row=" line
    }
' "$SWEEP_CSV" >"$BEST_TXT"

cat "$SWEEP_CSV"
cat "$BEST_TXT"
