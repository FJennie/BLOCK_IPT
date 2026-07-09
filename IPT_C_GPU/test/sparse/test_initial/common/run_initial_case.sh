#!/usr/bin/env bash
set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
CASE_ENV="${1:?usage: run_initial_case.sh CASE_ENV K}"
K_VALUE="${2:?usage: run_initial_case.sh CASE_ENV K}"

source "$CASE_ENV"

: "${IPT_MATRIX_ID:?IPT_MATRIX_ID is required by case env}"
: "${IPT_MOLECULE_ATOMS:?IPT_MOLECULE_ATOMS is required by case env}"

PRIMME_ROOT="${PRIMME_ROOT:-/fs1/home/nudt_liujie/ftt/primme-3.2.3}"
PYTHON_ROOT="${PYTHON_ROOT:-/fs1/software/python/3.8_miniconda_4.10.3}"
PYSCF_SITE_PACKAGES="${PYSCF_SITE_PACKAGES:-/fs1/software/hpcsystem/THL7/software/pyscf/2.0.1-py3.8/lib/python3.8/site-packages}"
COMMON_DIR="$ROOT/test/sparse/test_initial/common"
CASE_DIR="$(cd "$(dirname "$CASE_ENV")" && pwd)"
RESULT_BASE="${IPT_RESULT_BASE:-$ROOT/results/sparse/test_initial}"
LOG_BASE="${IPT_LOG_BASE:-$ROOT/logs/sparse/test_initial}"
RESULT_DIR="$RESULT_BASE/$IPT_MATRIX_ID/K$K_VALUE"
LOG_DIR="$LOG_BASE/$IPT_MATRIX_ID/K$K_VALUE"
BUILD_DIR="${IPT_BENCH_BUILD_DIR:-$COMMON_DIR/.build}"
BENCH_BIN="${IPT_BENCH_BIN:-$BUILD_DIR/benchmark_initial_molecule_${SLURM_JOB_ID:-manual}_$$}"
RUN_LOG="$LOG_DIR/${IPT_MATRIX_ID}_K${K_VALUE}_${SLURM_JOB_ID:-manual}.log"

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$BUILD_DIR"
ln -sfn "$RUN_LOG" "$LOG_DIR/latest.log"
exec > >(tee "$RUN_LOG") 2>&1

if [ -f /fs1/software/modules/4.2.1-gcc8.4.1/init/bash ]; then
    source /fs1/software/modules/4.2.1-gcc8.4.1/init/bash
elif [ -f /etc/profile.d/module.sh ]; then
    source /etc/profile.d/module.sh
fi
module unload CUDA/12.8.1 2>/dev/null || true
module unload CUDA/12.2 2>/dev/null || true
module load GCC/12.2.0
module load CUDA/12.2

OPENBLAS_LIB="${OPENBLAS_LIB:-/fs1/software/cp2k/8.1-gcc8.4-openmpi/lib/libopenblas.so.0}"
if [ ! -f "$OPENBLAS_LIB" ]; then OPENBLAS_LIB="/usr/lib64/libopenblas.so.0"; fi
if [ ! -f "$OPENBLAS_LIB" ]; then OPENBLAS_LIB="/lib64/libopenblas.so.0"; fi
OPENBLAS_DIR="$(dirname "$OPENBLAS_LIB")"
GCC_LIB_DIR="${GCC_LIB_DIR:-/fs1/software/gcc/12.2.0/lib64}"
ln -sfn "$OPENBLAS_LIB" "$COMMON_DIR/libopenblas.so"

export IPT_C_ROOT="$ROOT"
export IPT_C_RESULTS_DIR="$RESULT_DIR"
export IPT_C_LOG_DIR="$LOG_DIR"
export IPT_MATRIX_CACHE="${IPT_MATRIX_CACHE:-$CASE_DIR/${IPT_MATRIX_ID}_csc.bin}"
export IPT_LOAD_MATRIX="${IPT_LOAD_MATRIX:-1}"
export IPT_SAVE_MATRIX="${IPT_SAVE_MATRIX:-1}"
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
export IPT_DUMP_DIAG_GAPS="${IPT_DUMP_DIAG_GAPS:-0}"
export IPT_DUMP_COUPLING_GAP="${IPT_DUMP_COUPLING_GAP:-0}"
export IPT_DUMP_RESIDUAL_SUPPORT="${IPT_DUMP_RESIDUAL_SUPPORT:-0}"
export RUN_PRIMME="${RUN_PRIMME:-1}"
export RUN_IPT="${RUN_IPT:-1}"
export RUN_WARMUP="${RUN_WARMUP:-0}"
export PRIMME_MAX_MATVECS="${PRIMME_MAX_MATVECS:-200000}"
export IPT_EST_NNZ_PER_COL="${IPT_EST_NNZ_PER_COL:-1200}"
export IPT_CUDA_ARCH="${IPT_CUDA_ARCH:-sm_70}"
export PYSCF_SITE_PACKAGES
export PYTHONPATH="$PYSCF_SITE_PACKAGES:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$GCC_LIB_DIR:$PYTHON_ROOT/lib:$PYSCF_SITE_PACKAGES/numpy.libs:$PYSCF_SITE_PACKAGES/scipy.libs:$PYSCF_SITE_PACKAGES/h5py.libs:$PYSCF_SITE_PACKAGES/pyscf/lib:$OPENBLAS_DIR:$COMMON_DIR:${LD_LIBRARY_PATH:-}"

echo "===== test_initial $IPT_MATRIX_ID K=$K_VALUE ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "RESULT_DIR=$RESULT_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "CASE_ENV=$CASE_ENV"
echo "IPT_MATRIX_CACHE=$IPT_MATRIX_CACHE"
echo "IPT_MOLECULE_BASIS=${IPT_MOLECULE_BASIS:-sto-6g}"
echo "IPT_ACTIVE_ELECTRONS=${IPT_ACTIVE_ELECTRONS:-0}"
echo "IPT_ACTIVE_ORBITALS=${IPT_ACTIVE_ORBITALS:-0}"
echo "IPT_CUDA_ARCH=$IPT_CUDA_ARCH"
echo "IPT_MAX_LIFT_BLOCK_SIZE=${IPT_MAX_LIFT_BLOCK_SIZE:-default_64}"
echo "RUN_PRIMME=$RUN_PRIMME"
echo "RUN_IPT=$RUN_IPT"
module list 2>&1
nvcc --version
nvidia-smi || true

if [ "${IPT_SKIP_BUILD:-0}" = "1" ]; then
    echo "IPT_SKIP_BUILD=1; using BENCH_BIN=$BENCH_BIN"
else
    echo "===== build ====="
    set -x
    nvcc -O3 -std=c++14 -lineinfo -arch="$IPT_CUDA_ARCH" \
        -I"$PRIMME_ROOT/include" \
        -I"$PYTHON_ROOT/include/python3.8" \
        "$COMMON_DIR/benchmark_initial_molecule.cu" \
        "$PRIMME_ROOT/lib/libprimme.a" \
        -o "$BENCH_BIN" \
        -lcublas -lcusparse -lcusolver -lcudart \
        -L"$PYTHON_ROOT/lib" -lpython3.8 \
        -L"$COMMON_DIR" -lopenblas \
        -lpthread -ldl -lutil -lrt \
        -Xlinker -rpath -Xlinker "$GCC_LIB_DIR" \
        -Xlinker -rpath -Xlinker "$PYTHON_ROOT/lib" \
        -Xlinker -rpath -Xlinker "$OPENBLAS_DIR" \
        -lgfortran "$GCC_LIB_DIR/libstdc++.so" -lm
    set +x
fi

echo "===== run ====="
LOCK_FILE="$CASE_DIR/${IPT_MATRIX_ID}.run.lock"
if command -v flock >/dev/null 2>&1; then
    flock "$LOCK_FILE" "$BENCH_BIN"
else
    "$BENCH_BIN"
fi
status=$?
echo "finished at: $(date)"
exit "$status"
