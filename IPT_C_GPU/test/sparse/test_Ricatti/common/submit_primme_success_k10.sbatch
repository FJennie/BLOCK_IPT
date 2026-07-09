#!/usr/bin/env bash
#SBATCH --job-name=primme_k10
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
MANIFEST="${1:-$ROOT/test/sparse/test_Ricatti/common/cases_success_k10.txt}"
COMMON_DIR="$ROOT/test/sparse/test_initial/common"
BUILD_DIR="${IPT_BENCH_BUILD_DIR:-$ROOT/test/sparse/test_Ricatti/common/.build}"
BENCH_BIN="${IPT_BENCH_BIN:-$BUILD_DIR/benchmark_primme_success_${SLURM_JOB_ID:-manual}_$$}"
PRIMME_ROOT="${PRIMME_ROOT:-/fs1/home/nudt_liujie/ftt/primme-3.2.3}"
PYTHON_ROOT="${PYTHON_ROOT:-/fs1/software/python/3.8_miniconda_4.10.3}"

mkdir -p "$BUILD_DIR" "$ROOT/logs/sparse/test_Ricatti_primme_compare/campaign"

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
export IPT_RESULT_BASE="$ROOT/results/sparse/test_Ricatti_primme_compare"
export IPT_LOG_BASE="$ROOT/logs/sparse/test_Ricatti_primme_compare"
export RUN_IPT=0
export RUN_PRIMME=1
export RUN_WARMUP=0
export IPT_REPEATS=1
export IPT_TOL=1e-12
export IPT_SKIP_BUILD=1
export IPT_BENCH_BIN="$BENCH_BIN"
export LD_LIBRARY_PATH="$GCC_LIB_DIR:$PYTHON_ROOT/lib:$OPENBLAS_DIR:$COMMON_DIR:${LD_LIBRARY_PATH:-}"

echo "===== build GPU PRIMME benchmark ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "MANIFEST=$MANIFEST"
echo "BENCH_BIN=$BENCH_BIN"
set -x
nvcc -O3 -std=c++14 -lineinfo -arch="${IPT_CUDA_ARCH:-sm_70}" \
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

echo "===== run GPU PRIMME cases ====="
while read -r molecule k_value; do
    if [ -z "${molecule:-}" ] || [[ "$molecule" == \#* ]]; then
        continue
    fi
    case_env="$ROOT/test/sparse/test_Ricatti/$molecule/case.env"
    echo "===== PRIMME case $molecule K=$k_value ====="
    bash "$COMMON_DIR/run_initial_case.sh" "$case_env" "$k_value"
done < "$MANIFEST"

echo "finished PRIMME campaign at: $(date)"
