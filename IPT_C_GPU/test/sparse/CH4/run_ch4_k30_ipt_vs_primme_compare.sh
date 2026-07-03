#!/usr/bin/env bash
#SBATCH --job-name=ch4_k30_cmp
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/ch4_k30_ipt_vs_primme_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/ch4_k30_ipt_vs_primme_%j.err

set -u -o pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
TEST_DIR="$ROOT/test/sparse/CH4"
CACHE="${IPT_CH4_MATRIX_CACHE:-$ROOT/results/sparse/CH4_K30_debug/ch4_sto6g_fci_15876_csc.bin}"
TAG="${COMPARE_TAG:-${SLURM_JOB_ID:-manual}}"
RESULT_BASE="$ROOT/results/sparse/CH4_K30_debug/ipt_vs_primme_k30_v1"
LOG_BASE="$ROOT/logs/sparse/CH4_K30_dubug/ipt_vs_primme_k30_v1"
RESULT_DIR="$RESULT_BASE/$TAG"
LOG_DIR="$LOG_BASE/$TAG"
IPT_RESULT_DIR="$RESULT_DIR/ipt_block_active_davidson"
PRIMME_RESULT_DIR="$RESULT_DIR/primme_jdqmr_etol"
IPT_LOG_DIR="$LOG_DIR/ipt_block_active_davidson"
PRIMME_LOG_DIR="$LOG_DIR/primme_jdqmr_etol"
COMPARE_CSV="$RESULT_DIR/ch4_k30_ipt_vs_primme_compare.csv"
COMPARE_SUMMARY="$RESULT_DIR/ch4_k30_ipt_vs_primme_compare_summary.txt"
MAIN_LOG="$LOG_DIR/ch4_k30_ipt_vs_primme_compare_${TAG}.log"
IPT_RUNNER="$TEST_DIR/run_ch4_sto6g_fci_15876_ipt.sh"
PRIMME_RUNNER="$TEST_DIR/run_ch4_primme_k30.sh"
IPT_COMPARE_DENOM_CLIP="${IPT_COMPARE_DENOM_CLIP:-3e-9}"
IPT_COMPARE_EXTRA_STEPS="${IPT_COMPARE_EXTRA_STEPS:-20}"
IPT_COMPARE_ORTHO_REPEATS="${IPT_COMPARE_ORTHO_REPEATS:-2}"

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$IPT_RESULT_DIR" "$PRIMME_RESULT_DIR" \
    "$IPT_LOG_DIR" "$PRIMME_LOG_DIR"
exec > >(tee "$MAIN_LOG") 2>&1

echo "===== CH4 k=30 IPT vs PRIMME solve-only comparison ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "CACHE=$CACHE"
echo "RESULT_DIR=$RESULT_DIR"
echo "LOG_DIR=$LOG_DIR"

if [ ! -f "$CACHE" ]; then
    echo "missing matrix cache: $CACHE" >&2
    exit 2
fi

echo "===== IPT: repaired block-active selective Davidson parameters ====="
echo "IPT_MAXITER=200"
echo "IPT_BLOCK_CLUSTER_OVERSAMPLE=3"
echo "IPT_BLOCK_CLUSTER_QR=1"
echo "IPT_DAVIDSON_STEPS=30"
echo "IPT_DAVIDSON_EXTRA_STEPS=$IPT_COMPARE_EXTRA_STEPS"
echo "IPT_DAVIDSON_ACTIVE_MAX=2"
echo "IPT_DAVIDSON_FORCE_ACTIVE_PAIRS=28,29"
echo "IPT_DAVIDSON_BLOCK_ACTIVE=1"
echo "IPT_DAVIDSON_DENOM_CLIP=$IPT_COMPARE_DENOM_CLIP"
echo "IPT_DAVIDSON_ORTHO_REPEATS=$IPT_COMPARE_ORTHO_REPEATS"
echo "IPT_DAVIDSON_USE_BEST_SO_FAR=${IPT_DAVIDSON_USE_BEST_SO_FAR:-1}"

set +e
(
    export IPT_C_RESULTS_DIR="$IPT_RESULT_DIR"
    export IPT_C_LOG_DIR="$IPT_LOG_DIR"
    export IPT_CH4_MATRIX_CACHE="$CACHE"
    export IPT_LOAD_MATRIX=1
    export IPT_SAVE_MATRIX=0
    export IPT_K=30
    export IPT_REPEATS=1
    export IPT_TOL=1e-12
    export IPT_MAXITER=200
    export IPT_BLOCK_CLUSTER_MAXITER=200
    export IPT_BLOCK_CLUSTER_TOL=0
    export IPT_BLOCK_CLUSTER=1
    export IPT_BLOCK_CLUSTER_OVERSAMPLE=3
    export IPT_BLOCK_CLUSTER_QR=1
    export IPT_BLOCK_CLUSTER_ADAPTIVE=0
    export IPT_DAVIDSON_ENRICH=1
    export IPT_DAVIDSON_STEPS=30
    export IPT_DAVIDSON_EXTRA_STEPS="$IPT_COMPARE_EXTRA_STEPS"
    export IPT_DAVIDSON_ACTIVE_MAX=2
    export IPT_DAVIDSON_SELECT_TOL=1e-12
    export IPT_DAVIDSON_PROTECT_TOL=1e-10
    export IPT_DAVIDSON_CONVERGED_TOL=1e-13
    export IPT_DAVIDSON_ACTIVE_TOL=1e-13
    export IPT_DAVIDSON_FORCE_ACTIVE_PAIRS=28,29
    export IPT_DAVIDSON_BLOCK_ACTIVE=1
    export IPT_DAVIDSON_DENOM_CLIP="$IPT_COMPARE_DENOM_CLIP"
    export IPT_DAVIDSON_ACCEPT_ONLY_IF_IMPROVES=1
    export IPT_DAVIDSON_USE_BEST_SO_FAR="${IPT_DAVIDSON_USE_BEST_SO_FAR:-1}"
    export IPT_DAVIDSON_ORTHO_REPEATS="$IPT_COMPARE_ORTHO_REPEATS"
    export IPT_DAVIDSON_RESTART_EVERY=20
    export IPT_DAVIDSON_RESTART_KEEP_EXTRA=5
    export IPT_DUMP_PAIR_RESIDUALS=1
    export IPT_DUMP_RESIDUAL_SUPPORT=0
    export IPT_JD_LOCAL_CORRECTION=0
    export RUN_PRIMME=0
    export RUN_WARMUP=0
    bash "$IPT_RUNNER"
)
ipt_exit=$?
echo "IPT exit code: $ipt_exit"

echo "===== PRIMME: JDQMR_ETol relative residual target ====="
(
    export IPT_CH4_MATRIX_CACHE="$CACHE"
    export PRIMME_CH4_OUTPUT_DIR="$PRIMME_RESULT_DIR"
    export PRIMME_CH4_LOG_DIR="$PRIMME_LOG_DIR"
    export PRIMME_K=30
    export PRIMME_REPEATS=1
    export PRIMME_RELATIVE_EIGEN_RESIDUAL_TOL=5e-13
    export PRIMME_REQUIRED_RELATIVE_EIGEN_RESIDUAL=1e-12
    export PRIMME_TOL="$PRIMME_RELATIVE_EIGEN_RESIDUAL_TOL"
    export PRIMME_MAX_MATVECS=200000
    export PRIMME_MAX_BASIS_SIZE=160
    export PRIMME_MIN_RESTART_SIZE=80
    export PRIMME_MAX_BLOCK_SIZE=8
    export PRIMME_LOCKING=1
    export PRIMME_METHOD=PRIMME_JDQMR_ETol
    export PRIMME_PRINT_LEVEL=0
    bash "$PRIMME_RUNNER"
)
primme_exit=$?
echo "PRIMME exit code: $primme_exit"
set -e

export RESULT_DIR LOG_DIR IPT_RESULT_DIR PRIMME_RESULT_DIR IPT_LOG_DIR \
    PRIMME_LOG_DIR COMPARE_CSV COMPARE_SUMMARY ipt_exit primme_exit CACHE \
    IPT_COMPARE_DENOM_CLIP IPT_COMPARE_EXTRA_STEPS IPT_COMPARE_ORTHO_REPEATS

python3 - <<'PY'
import csv
import math
import os
import re
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR"])
ipt_result_dir = Path(os.environ["IPT_RESULT_DIR"])
primme_result_dir = Path(os.environ["PRIMME_RESULT_DIR"])
compare_csv = Path(os.environ["COMPARE_CSV"])
compare_summary = Path(os.environ["COMPARE_SUMMARY"])
ipt_exit = int(os.environ["ipt_exit"])
primme_exit = int(os.environ["primme_exit"])

def read_first_csv(path):
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None

def parse_ipt_solve_time(log_dir):
    log_path = None
    for candidate in sorted(Path(log_dir).glob("ch4_sto6g_fci_15876_ipt_*.log")):
        log_path = candidate
    if log_path is None:
        latest = Path(log_dir) / "latest_ch4_sto6g_fci_15876_ipt.log"
        if latest.exists():
            log_path = latest
    if log_path is None:
        return "", ""
    pattern = re.compile(r"repeat\s+1\s+IPT_GPU_BLOCK.*?\bsolve=([0-9.eE+-]+)")
    text = log_path.read_text(errors="replace")
    matches = pattern.findall(text)
    if not matches:
        return "", str(log_path)
    return matches[-1], str(log_path)

ipt_trial = read_first_csv(
    ipt_result_dir / "ch4_sto6g_fci_15876_ipt_relative_trials.csv")
primme_timing = read_first_csv(
    primme_result_dir / "ch4_sto6g_fci_15876_primme_k30_timing.csv")
ipt_solve_time, ipt_solve_log = parse_ipt_solve_time(os.environ["IPT_LOG_DIR"])

rows = []
if ipt_trial:
    rows.append({
        "method": "IPT_block_active_selective_Davidson",
        "exit_code": str(ipt_exit),
        "status": ipt_trial.get("status", ""),
        "requested_k": ipt_trial.get("requested_k", "30"),
        "returned_k": ipt_trial.get("returned_k", ""),
        "basis_cols": ipt_trial.get("basis_cols", ""),
        "iterations": ipt_trial.get("iterations", ""),
        "matvecs": "",
        "solve_time_sec": ipt_solve_time or ipt_trial.get("time_total_sec", ""),
        "solve_time_source": "ipt_log_solve_field" if ipt_solve_time else "trial_csv_time_total_sec_fallback",
        "max_relative_eigen_residual":
            ipt_trial.get("max_relative_eigen_residual", ""),
        "max_relative_eigen_residual_index":
            ipt_trial.get("max_relative_eigen_residual_index", ""),
        "relative_fixed_point_residual":
            ipt_trial.get("relative_fixed_point_residual", ""),
        "best_so_far_updated": ipt_trial.get("best_so_far_updated", ""),
        "best_so_far_step": ipt_trial.get("best_so_far_step", ""),
        "best_so_far_source": ipt_trial.get("best_so_far_source", ""),
        "best_so_far_max_residual":
            ipt_trial.get("best_so_far_max_residual", ""),
        "final_returned_from_best_so_far":
            ipt_trial.get("final_returned_from_best_so_far", ""),
        "pair28_best_residual": ipt_trial.get("pair28_best_residual", ""),
        "pair29_best_residual": ipt_trial.get("pair29_best_residual", ""),
        "passed_relative_threshold":
            "1" if ipt_trial.get("status", "") == "success" else "0",
        "result_dir": str(ipt_result_dir),
        "log_dir": os.environ["IPT_LOG_DIR"],
    })
else:
    rows.append({
        "method": "IPT_block_active_selective_Davidson",
        "exit_code": str(ipt_exit),
        "status": "missing_output",
        "requested_k": "30",
        "returned_k": "",
        "basis_cols": "",
        "iterations": "",
        "matvecs": "",
        "solve_time_sec": "",
        "solve_time_source": "",
        "max_relative_eigen_residual": "",
        "max_relative_eigen_residual_index": "",
        "relative_fixed_point_residual": "",
        "best_so_far_updated": "",
        "best_so_far_step": "",
        "best_so_far_source": "",
        "best_so_far_max_residual": "",
        "final_returned_from_best_so_far": "",
        "pair28_best_residual": "",
        "pair29_best_residual": "",
        "passed_relative_threshold": "0",
        "result_dir": str(ipt_result_dir),
        "log_dir": os.environ["IPT_LOG_DIR"],
    })

if primme_timing:
    rows.append({
        "method": "PRIMME_JDQMR_ETol",
        "exit_code": str(primme_exit),
        "status": "success" if primme_timing.get("passed_relative_threshold") == "1"
                  and primme_timing.get("status") == "0" else "failed",
        "requested_k": "30",
        "returned_k": primme_timing.get("returned_k", ""),
        "basis_cols": "",
        "iterations": primme_timing.get("outer_iterations", ""),
        "matvecs": primme_timing.get("matvecs", ""),
        "solve_time_sec": primme_timing.get("solve_time_sec", ""),
        "solve_time_source": "primme_timing_csv_solve_time_sec",
        "max_relative_eigen_residual":
            primme_timing.get("max_relative_eigen_residual", ""),
        "max_relative_eigen_residual_index":
            primme_timing.get("max_relative_eigen_residual_index", ""),
        "relative_fixed_point_residual": "",
        "best_so_far_updated": "",
        "best_so_far_step": "",
        "best_so_far_source": "",
        "best_so_far_max_residual": "",
        "final_returned_from_best_so_far": "",
        "pair28_best_residual": "",
        "pair29_best_residual": "",
        "passed_relative_threshold":
            primme_timing.get("passed_relative_threshold", ""),
        "result_dir": str(primme_result_dir),
        "log_dir": os.environ["PRIMME_LOG_DIR"],
    })
else:
    rows.append({
        "method": "PRIMME_JDQMR_ETol",
        "exit_code": str(primme_exit),
        "status": "missing_output",
        "requested_k": "30",
        "returned_k": "",
        "basis_cols": "",
        "iterations": "",
        "matvecs": "",
        "solve_time_sec": "",
        "solve_time_source": "",
        "max_relative_eigen_residual": "",
        "max_relative_eigen_residual_index": "",
        "relative_fixed_point_residual": "",
        "best_so_far_updated": "",
        "best_so_far_step": "",
        "best_so_far_source": "",
        "best_so_far_max_residual": "",
        "final_returned_from_best_so_far": "",
        "pair28_best_residual": "",
        "pair29_best_residual": "",
        "passed_relative_threshold": "0",
        "result_dir": str(primme_result_dir),
        "log_dir": os.environ["PRIMME_LOG_DIR"],
    })

fieldnames = [
    "method", "exit_code", "status", "requested_k", "returned_k",
    "basis_cols", "iterations", "matvecs", "solve_time_sec",
    "solve_time_source", "max_relative_eigen_residual",
    "max_relative_eigen_residual_index",
    "relative_fixed_point_residual", "best_so_far_updated",
    "best_so_far_step", "best_so_far_source",
    "best_so_far_max_residual", "final_returned_from_best_so_far",
    "pair28_best_residual", "pair29_best_residual",
    "passed_relative_threshold",
    "result_dir", "log_dir",
]
with compare_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def as_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan

ipt_time = as_float(rows[0]["solve_time_sec"])
primme_time = as_float(rows[1]["solve_time_sec"])
speedup = ipt_time / primme_time if ipt_time > 0 and primme_time > 0 else math.nan

with compare_summary.open("w") as f:
    f.write("matrix_source=cache\n")
    f.write(f"cache_path={os.environ['CACHE']}\n")
    f.write("requested_k=30\n")
    f.write("convergence_metric=max_relative_eigen_residual\n")
    f.write("convergence_threshold=1e-12\n")
    f.write("timing_metric=solve_time_sec\n")
    f.write("timing_scope=solver_only_excludes_cache_read_h2d_setup_d2h_residual_recompute\n")
    f.write(
        "ipt_parameters=iter200_OS3_QR_selective_Davidson30_"
        "force28_29_block_active_"
        f"extra{os.environ['IPT_COMPARE_EXTRA_STEPS']}_"
        f"clip{os.environ['IPT_COMPARE_DENOM_CLIP']}_"
        f"ortho{os.environ['IPT_COMPARE_ORTHO_REPEATS']}_"
        "restart20_keep5\n")
    f.write("primme_parameters=JDQMR_ETol_relative_tol5e-13_required1e-12_maxBasis160_minRestart80_block8\n")
    for row in rows:
        f.write(
            f"method={row['method']} status={row['status']} "
            f"exit_code={row['exit_code']} returned_k={row['returned_k']} "
            f"solve_time_sec={row['solve_time_sec']} "
            f"solve_time_source={row['solve_time_source']} "
            f"max_relative_eigen_residual={row['max_relative_eigen_residual']} "
            f"max_relative_eigen_residual_index={row['max_relative_eigen_residual_index']} "
            f"best_so_far_updated={row['best_so_far_updated']} "
            f"best_so_far_step={row['best_so_far_step']} "
            f"best_so_far_source={row['best_so_far_source']} "
            f"best_so_far_max_residual={row['best_so_far_max_residual']} "
            f"final_returned_from_best_so_far={row['final_returned_from_best_so_far']} "
            f"pair28_best_residual={row['pair28_best_residual']} "
            f"pair29_best_residual={row['pair29_best_residual']} "
            f"passed_relative_threshold={row['passed_relative_threshold']}\n")
    f.write(f"primme_speedup_vs_ipt={speedup:.17g}\n")
    f.write(f"ipt_solve_log={ipt_solve_log}\n")
    f.write(f"compare_csv={compare_csv}\n")
    f.write(f"result_dir={result_dir}\n")
    f.write(f"log_dir={os.environ['LOG_DIR']}\n")

print(compare_csv.read_text())
print(compare_summary.read_text())
PY

echo "wrote $COMPARE_CSV"
echo "wrote $COMPARE_SUMMARY"
echo "logs under $LOG_DIR"
echo "finished at: $(date)"

if [ "$ipt_exit" -ne 0 ] || [ "$primme_exit" -ne 0 ]; then
    exit 1
fi
