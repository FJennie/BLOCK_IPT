#!/usr/bin/env bash
set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
LOG_DIR="${IPT_C_LOG_DIR:-$ROOT/logs/sparse}"
RESULTS_DIR="${IPT_C_RESULTS_DIR:-$ROOT/results/sparse}"
TEST_DIR="$ROOT/test/sparse"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$TEST_DIR"

RUN_LOG="$LOG_DIR/sparse_tests_${SLURM_JOB_ID:-manual}.log"
ln -sfn "$RUN_LOG" "$LOG_DIR/latest_sparse_tests.log"
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

cd "$ROOT"

echo "===== IPT_C_GPU sparse CUDA tests ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "ROOT=$ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "RESULTS_DIR=$RESULTS_DIR"
module list 2>&1
echo

nvcc --version
echo
nvidia-smi || true
echo

echo "===== build ====="
set -x
nvcc -O3 -std=c++14 -lineinfo -arch=sm_80 \
    "$TEST_DIR/runtests_sparse.cu" \
    -o "$TEST_DIR/runtests_sparse_cuda" \
    -lcublas -lcusparse -lcusolver
set +x

echo
echo "===== run ====="
IPT_C_RESULTS_DIR="$RESULTS_DIR" \
IPT_SPARSE_RESULTS_DIR="$RESULTS_DIR" \
    "$TEST_DIR/runtests_sparse_cuda"
status=$?

echo
echo "===== output files ====="
ls -lh "$LOG_DIR" "$RESULTS_DIR" || true
echo "finished at: $(date)"

exit "$status"
