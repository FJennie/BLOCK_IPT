#!/usr/bin/env bash
#SBATCH --job-name=ch4_primme_k30
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/test/sparse/CH4/primme_k30_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/test/sparse/CH4/primme_k30_%j.err

set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
PRIMME_ROOT="${PRIMME_ROOT:-/fs1/home/nudt_liujie/ftt/primme-3.2.3}"
PYTHON_ROOT="${PYTHON_ROOT:-/fs1/software/python/3.8_miniconda_4.10.3}"
PYSCF_SITE_PACKAGES="${PYSCF_SITE_PACKAGES:-/fs1/software/hpcsystem/THL7/software/pyscf/2.0.1-py3.8/lib/python3.8/site-packages}"
TEST_DIR="$ROOT/test/sparse/CH4"
SOURCE="$TEST_DIR/benchmark_ch4_sto6g_fci_15876_primme_k30.cu"
BINARY="$TEST_DIR/benchmark_ch4_sto6g_fci_15876_primme_k30_cuda"
OUTPUT_DIR="${PRIMME_CH4_OUTPUT_DIR:-$TEST_DIR/primme_k30_results}"
LOG_DIR="${PRIMME_CH4_LOG_DIR:-$OUTPUT_DIR}"
RUN_LOG="$LOG_DIR/ch4_primme_k30_${SLURM_JOB_ID:-manual}.log"
CACHE="${IPT_CH4_MATRIX_CACHE:-$ROOT/results/sparse/CH4_K30_debug/ch4_sto6g_fci_15876_csc.bin}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
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
if [ ! -f "$OPENBLAS_LIB" ]; then
    OPENBLAS_LIB="/usr/lib64/libopenblas.so.0"
fi
if [ ! -f "$OPENBLAS_LIB" ]; then
    OPENBLAS_LIB="/lib64/libopenblas.so.0"
fi
OPENBLAS_DIR="$(dirname "$OPENBLAS_LIB")"
GCC_LIB_DIR="${GCC_LIB_DIR:-/fs1/software/gcc/12.2.0/lib64}"
ln -sfn "$OPENBLAS_LIB" "$TEST_DIR/libopenblas.so"
export LD_LIBRARY_PATH="$GCC_LIB_DIR:$PYTHON_ROOT/lib:$PYSCF_SITE_PACKAGES/numpy.libs:$PYSCF_SITE_PACKAGES/scipy.libs:$PYSCF_SITE_PACKAGES/h5py.libs:$PYSCF_SITE_PACKAGES/pyscf/lib:$OPENBLAS_DIR:$TEST_DIR:${LD_LIBRARY_PATH:-}"
export IPT_CH4_MATRIX_CACHE="$CACHE"
export PRIMME_CH4_OUTPUT_DIR="$OUTPUT_DIR"
export PRIMME_CH4_LOG_DIR="$LOG_DIR"
export PRIMME_K="${PRIMME_K:-30}"
export PRIMME_REPEATS="${PRIMME_REPEATS:-1}"
export PRIMME_RELATIVE_EIGEN_RESIDUAL_TOL="${PRIMME_RELATIVE_EIGEN_RESIDUAL_TOL:-5e-13}"
export PRIMME_REQUIRED_RELATIVE_EIGEN_RESIDUAL="${PRIMME_REQUIRED_RELATIVE_EIGEN_RESIDUAL:-1e-12}"
export PRIMME_TOL="${PRIMME_TOL:-$PRIMME_RELATIVE_EIGEN_RESIDUAL_TOL}"
export PRIMME_MAX_MATVECS="${PRIMME_MAX_MATVECS:-200000}"
export PRIMME_MAX_BASIS_SIZE="${PRIMME_MAX_BASIS_SIZE:-160}"
export PRIMME_MIN_RESTART_SIZE="${PRIMME_MIN_RESTART_SIZE:-80}"
export PRIMME_MAX_BLOCK_SIZE="${PRIMME_MAX_BLOCK_SIZE:-8}"
export PRIMME_LOCKING="${PRIMME_LOCKING:-1}"
export PRIMME_METHOD="${PRIMME_METHOD:-PRIMME_JDQMR_ETol}"
export PRIMME_PRINT_LEVEL="${PRIMME_PRINT_LEVEL:-0}"

echo "===== CH4/STO-6G FCI PRIMME k=30 ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PRIMME_K=$PRIMME_K"
echo "PRIMME_REPEATS=$PRIMME_REPEATS"
echo "PRIMME_TOL=$PRIMME_TOL"
echo "PRIMME_RELATIVE_EIGEN_RESIDUAL_TOL=$PRIMME_RELATIVE_EIGEN_RESIDUAL_TOL"
echo "PRIMME_REQUIRED_RELATIVE_EIGEN_RESIDUAL=$PRIMME_REQUIRED_RELATIVE_EIGEN_RESIDUAL"
echo "PRIMME_MAX_MATVECS=$PRIMME_MAX_MATVECS"
echo "PRIMME_MAX_BASIS_SIZE=$PRIMME_MAX_BASIS_SIZE"
echo "PRIMME_MIN_RESTART_SIZE=$PRIMME_MIN_RESTART_SIZE"
echo "PRIMME_MAX_BLOCK_SIZE=$PRIMME_MAX_BLOCK_SIZE"
echo "PRIMME_LOCKING=$PRIMME_LOCKING"
echo "PRIMME_METHOD=$PRIMME_METHOD"
echo "PRIMME_PRINT_LEVEL=$PRIMME_PRINT_LEVEL"
echo "IPT_CH4_MATRIX_CACHE=$IPT_CH4_MATRIX_CACHE"
echo "PRIMME_CH4_OUTPUT_DIR=$PRIMME_CH4_OUTPUT_DIR"
echo "PRIMME_CH4_LOG_DIR=$PRIMME_CH4_LOG_DIR"

if [ "${PRIMME_SKIP_BUILD:-0}" != "1" ]; then
    nvcc -O3 -std=c++14 -lineinfo -arch=sm_80 \
        -I"$PRIMME_ROOT/include" \
        -I"$PYTHON_ROOT/include/python3.8" \
        "$SOURCE" \
        "$PRIMME_ROOT/lib/libprimme.a" \
        -o "$BINARY" \
        -lcublas -lcusparse -lcusolver -lcudart \
        -L"$PYTHON_ROOT/lib" -lpython3.8 \
        -L"$TEST_DIR" -lopenblas \
        -lpthread -ldl -lutil -lrt \
        -Xlinker -rpath -Xlinker "$GCC_LIB_DIR" \
        -Xlinker -rpath -Xlinker "$PYTHON_ROOT/lib" \
        -Xlinker -rpath -Xlinker "$OPENBLAS_DIR" \
        -lgfortran "$GCC_LIB_DIR/libstdc++.so" -lm
fi

"$BINARY"
