#!/usr/bin/env bash
set -uo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
COMMON_DIR="$ROOT/test/sparse/test_initial/common"
MATRIX_REF="${1:?usage: run_initial_cache_case.sh MATRIX_DIR_OR_ID K [PARAM_SET]}"
K_VALUE="${2:?usage: run_initial_cache_case.sh MATRIX_DIR_OR_ID K [PARAM_SET]}"
PARAM_SET="${3:-${PARAM_SET:-default}}"

if [ -d "$MATRIX_REF" ]; then
    CASE_DIR="$(cd "$MATRIX_REF" && pwd)"
else
    CASE_DIR="$ROOT/test/sparse/test_initial/$MATRIX_REF"
fi
MATRIX_ID="$(basename "$CASE_DIR")"

RESULT_BASE="${IPT_RESULT_BASE:-$ROOT/results/sparse/test_initial}"
LOG_BASE="${IPT_LOG_BASE:-$ROOT/logs/sparse/test_initial}"
RESULT_DIR="$RESULT_BASE/$MATRIX_ID/K$K_VALUE/$PARAM_SET"
IPT_RESULT_DIR="$RESULT_DIR/ipt"
PRIMME_RESULT_DIR="$RESULT_DIR/gpu_primme"
LOG_DIR="$LOG_BASE/$MATRIX_ID/K$K_VALUE/$PARAM_SET"
RUN_LOG="$LOG_DIR/run_${MATRIX_ID}_K${K_VALUE}_${PARAM_SET}_${SLURM_JOB_ID:-manual}.log"
STATUS_FILE="$RESULT_DIR/run_status.tsv"

BENCH_BIN="${IPT_BENCH_BIN:-$COMMON_DIR/.build/benchmark_initial_campaign_sm70}"
GPU_PRIMME_BIN="${IPT_GPU_PRIMME_BIN:-$COMMON_DIR/.build/benchmark_initial_campaign_sm70_gpuprimme}"
RUN_GPU_PRIMME_ON_SUCCESS="${RUN_GPU_PRIMME_ON_SUCCESS:-1}"
CONVERGENCE_THRESHOLD="${CONVERGENCE_THRESHOLD:-1e-12}"

mkdir -p "$RESULT_DIR" "$IPT_RESULT_DIR" "$PRIMME_RESULT_DIR" "$LOG_DIR"
ln -sfn "$RUN_LOG" "$LOG_DIR/latest.log"
exec > >(tee "$RUN_LOG") 2>&1

echo "===== cache-only test_initial $MATRIX_ID K=$K_VALUE param=$PARAM_SET ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "CASE_DIR=$CASE_DIR"
echo "RESULT_DIR=$RESULT_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "BENCH_BIN=$BENCH_BIN"
echo "GPU_PRIMME_BIN=$GPU_PRIMME_BIN"

write_status() {
    local key="$1"
    local value="$2"
    printf "%s\t%s\n" "$key" "$value" >> "$STATUS_FILE"
}

: > "$STATUS_FILE"
write_status "matrix_id" "$MATRIX_ID"
write_status "k" "$K_VALUE"
write_status "param_set" "$PARAM_SET"
write_status "result_dir" "$RESULT_DIR"
write_status "log" "$RUN_LOG"

if [ ! -d "$CASE_DIR" ]; then
    echo "missing matrix directory: $CASE_DIR"
    write_status "ipt_status" "matrix_dir_missing"
    exit 0
fi

MATRIX_CACHE="$(find "$CASE_DIR" -maxdepth 1 -type f -name '*_csc.bin' | sort | head -n 1)"
if [ -z "$MATRIX_CACHE" ]; then
    echo "missing trusted matrix cache in $CASE_DIR"
    write_status "ipt_status" "matrix_cache_missing"
    exit 0
fi
write_status "matrix_cache" "$MATRIX_CACHE"
ls -lh "$MATRIX_CACHE"

if [ -f /fs1/software/modules/4.2.1-gcc8.4.1/init/bash ]; then
    # shellcheck disable=SC1091
    source /fs1/software/modules/4.2.1-gcc8.4.1/init/bash
elif [ -f /etc/profile.d/module.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/module.sh
fi
module unload CUDA/12.8.1 2>/dev/null || true
module unload CUDA/12.2 2>/dev/null || true
module load GCC/12.2.0
module load CUDA/12.2

PYTHON_ROOT="${PYTHON_ROOT:-/fs1/software/python/3.8_miniconda_4.10.3}"
PYSCF_SITE_PACKAGES="${PYSCF_SITE_PACKAGES:-/fs1/software/hpcsystem/THL7/software/pyscf/2.0.1-py3.8/lib/python3.8/site-packages}"
OPENBLAS_LIB="${OPENBLAS_LIB:-/fs1/software/cp2k/8.1-gcc8.4-openmpi/lib/libopenblas.so.0}"
if [ ! -f "$OPENBLAS_LIB" ]; then OPENBLAS_LIB="/usr/lib64/libopenblas.so.0"; fi
if [ ! -f "$OPENBLAS_LIB" ]; then OPENBLAS_LIB="/lib64/libopenblas.so.0"; fi
OPENBLAS_DIR="$(dirname "$OPENBLAS_LIB")"
GCC_LIB_DIR="${GCC_LIB_DIR:-/fs1/software/gcc/12.2.0/lib64}"
ln -sfn "$OPENBLAS_LIB" "$COMMON_DIR/libopenblas.so"

export PYTHONPATH="$PYSCF_SITE_PACKAGES:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$GCC_LIB_DIR:$PYTHON_ROOT/lib:$PYSCF_SITE_PACKAGES/numpy.libs:$PYSCF_SITE_PACKAGES/scipy.libs:$PYSCF_SITE_PACKAGES/h5py.libs:$PYSCF_SITE_PACKAGES/pyscf/lib:$OPENBLAS_DIR:$COMMON_DIR:${LD_LIBRARY_PATH:-}"

unset IPT_MOLECULE_ATOMS IPT_MOLECULE_BASIS IPT_MOLECULE_CHARGE IPT_MOLECULE_SPIN
unset IPT_ACTIVE_ELECTRONS IPT_ACTIVE_ORBITALS IPT_CHARGE IPT_SPIN

export IPT_MATRIX_ID="$MATRIX_ID"
export IPT_MATRIX_CACHE="$MATRIX_CACHE"
export IPT_LOAD_MATRIX=1
export IPT_SAVE_MATRIX=0
export IPT_REPEATS="${IPT_REPEATS:-1}"
export IPT_K="$K_VALUE"
export IPT_TOL="${IPT_TOL:-1e-12}"
export IPT_MAXITER="${IPT_MAXITER:-1000}"
export IPT_BLOCK_CLUSTER="${IPT_BLOCK_CLUSTER:-1}"
export IPT_BLOCK_CLUSTER_OVERSAMPLE="${IPT_BLOCK_CLUSTER_OVERSAMPLE:-3}"
export IPT_BLOCK_CLUSTER_QR="${IPT_BLOCK_CLUSTER_QR:-1}"
export IPT_BLOCK_CLUSTER_ADAPTIVE="${IPT_BLOCK_CLUSTER_ADAPTIVE:-0}"
export IPT_BLOCK_CLUSTER_MAXITER="${IPT_BLOCK_CLUSTER_MAXITER:-200}"
export IPT_BLOCK_CLUSTER_TOL="${IPT_BLOCK_CLUSTER_TOL:-0}"
export IPT_DAVIDSON_ENRICH="${IPT_DAVIDSON_ENRICH:-1}"
export IPT_DAVIDSON_STEPS="${IPT_DAVIDSON_STEPS:-30}"
export IPT_DAVIDSON_EXTRA_STEPS="${IPT_DAVIDSON_EXTRA_STEPS:-20}"
export IPT_DAVIDSON_CONVERGED_TOL="${IPT_DAVIDSON_CONVERGED_TOL:-1e-12}"
export IPT_DUMP_PAIR_RESIDUALS="${IPT_DUMP_PAIR_RESIDUALS:-1}"
export IPT_DUMP_DIAG_GAPS="${IPT_DUMP_DIAG_GAPS:-1}"
export IPT_DUMP_COUPLING_GAP="${IPT_DUMP_COUPLING_GAP:-1}"
export IPT_DUMP_RESIDUAL_SUPPORT="${IPT_DUMP_RESIDUAL_SUPPORT:-0}"
export PRIMME_MAX_MATVECS="${PRIMME_MAX_MATVECS:-200000}"
export IPT_EST_NNZ_PER_COL="${IPT_EST_NNZ_PER_COL:-1200}"
export IPT_CUDA_ARCH="${IPT_CUDA_ARCH:-sm_70}"

{
    echo "PARAM_SET=$PARAM_SET"
    echo "IPT_MATRIX_ID=$IPT_MATRIX_ID"
    echo "IPT_MATRIX_CACHE=$IPT_MATRIX_CACHE"
    echo "IPT_K=$IPT_K"
    echo "IPT_TOL=$IPT_TOL"
    echo "IPT_MAXITER=$IPT_MAXITER"
    echo "IPT_MAX_LIFT_BLOCK_SIZE=${IPT_MAX_LIFT_BLOCK_SIZE:-64_default}"
    echo "IPT_MAX_LIFT_ROUNDS=${IPT_MAX_LIFT_ROUNDS:-3_default}"
    echo "IPT_REL_GAP_TOL=${IPT_REL_GAP_TOL:-1e-12_default}"
    echo "IPT_COUPLING_GAP_RATIO_TOL=${IPT_COUPLING_GAP_RATIO_TOL:-0.2_default}"
    echo "IPT_COUPLING_ETA=${IPT_COUPLING_ETA:-0.02_default}"
    echo "IPT_BLOCK_CLUSTER_OVERSAMPLE=$IPT_BLOCK_CLUSTER_OVERSAMPLE"
    echo "IPT_BLOCK_CLUSTER_ADAPTIVE=$IPT_BLOCK_CLUSTER_ADAPTIVE"
    echo "IPT_BLOCK_CLUSTER_MAXITER=$IPT_BLOCK_CLUSTER_MAXITER"
    echo "IPT_DAVIDSON_STEPS=$IPT_DAVIDSON_STEPS"
    echo "IPT_DAVIDSON_EXTRA_STEPS=$IPT_DAVIDSON_EXTRA_STEPS"
    echo "IPT_DAVIDSON_CONVERGED_TOL=$IPT_DAVIDSON_CONVERGED_TOL"
    echo "PRIMME_MAX_MATVECS=$PRIMME_MAX_MATVECS"
    echo "timing_policy_ipt=compute_time_sec is Riccati/block fixed-point plus Rayleigh-Ritz plus Davidson; preparation/setup/transfer/result-copy excluded"
    echo "timing_policy_primme=compute_time_sec is cublas_dprimme only; GPU setup/transfer/result-copy excluded"
} > "$RESULT_DIR/params.env"

echo "===== environment ====="
module list 2>&1 || true
nvcc --version || true
nvidia-smi || true
cat "$RESULT_DIR/params.env"

echo "===== IPT run ====="
IPT_C_RESULTS_DIR="$IPT_RESULT_DIR" RUN_IPT=1 RUN_PRIMME=0 "$BENCH_BIN"
ipt_rc=$?
write_status "ipt_rc" "$ipt_rc"

IPT_SUMMARY="$IPT_RESULT_DIR/${MATRIX_ID}_summary.txt"
IPT_SUCCESS="$(python3 - "$IPT_SUMMARY" "$CONVERGENCE_THRESHOLD" <<'PY'
import math
import sys
from pathlib import Path

summary = Path(sys.argv[1])
threshold = float(sys.argv[2])
ok = False
if summary.is_file():
    for line in summary.read_text(errors="replace").splitlines():
        if "method=IPT_CUDA_BLOCK_CLUSTER" not in line:
            continue
        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
        residual = fields.get("max_relative_eigen_residual")
        try:
            residual_value = float(residual)
        except (TypeError, ValueError):
            residual_value = math.inf
        ok = fields.get("status") == "success" and residual_value <= threshold
print("1" if ok else "0")
PY
)"
write_status "ipt_success" "$IPT_SUCCESS"
write_status "ipt_summary" "$IPT_SUMMARY"

primme_rc="skipped"
if [ "$RUN_GPU_PRIMME_ON_SUCCESS" = "1" ] && [ "$IPT_SUCCESS" = "1" ]; then
    echo "===== GPU PRIMME run ====="
    IPT_C_RESULTS_DIR="$PRIMME_RESULT_DIR" RUN_IPT=0 RUN_PRIMME=1 "$GPU_PRIMME_BIN"
    primme_rc=$?
else
    echo "GPU PRIMME skipped; RUN_GPU_PRIMME_ON_SUCCESS=$RUN_GPU_PRIMME_ON_SUCCESS IPT_SUCCESS=$IPT_SUCCESS"
fi
write_status "primme_rc" "$primme_rc"
write_status "primme_summary" "$PRIMME_RESULT_DIR/${MATRIX_ID}_summary.txt"

echo "finished at: $(date)"
exit 0
