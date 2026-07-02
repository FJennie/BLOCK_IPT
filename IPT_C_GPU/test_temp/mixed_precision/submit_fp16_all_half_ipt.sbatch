#!/usr/bin/env bash
#SBATCH --job-name=fp16_all_half
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/mixed_precision/fp16_all_half_ipt_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/mixed_precision/fp16_all_half_ipt_%j.err

set -uo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
TEST_DIR="$ROOT/test/mixed_precision"
LOG_DIR="${IPT_C_LOG_DIR:-$ROOT/logs/mixed_precision}"
RESULTS_DIR="${IPT_C_RESULTS_DIR:-$ROOT/results/mixed_precision}"

mkdir -p "$TEST_DIR" "$LOG_DIR" "$RESULTS_DIR"

RUN_LOG="$LOG_DIR/fp16_all_half_ipt_${SLURM_JOB_ID:-manual}.log"
ln -sfn "$RUN_LOG" "$LOG_DIR/latest_fp16_all_half_ipt.log"
exec > >(tee "$RUN_LOG") 2>&1

if [ -f /fs1/software/modules/4.2.1-gcc8.4.1/init/bash ]; then
    source /fs1/software/modules/4.2.1-gcc8.4.1/init/bash
elif [ -f /etc/profile.d/module.sh ]; then
    source /etc/profile.d/module.sh
fi

module purge || true
module load GCC/12.2.0
module load CUDA/12.2

cd "$ROOT"

export IPT_C_ROOT="$ROOT"
export IPT_C_LOG_DIR="$LOG_DIR"
export IPT_C_RESULTS_DIR="$RESULTS_DIR"
export IPT_ALL_HALF_N="${IPT_ALL_HALF_N:-26000}"
export IPT_ALL_HALF_EPSILON="${IPT_ALL_HALF_EPSILON:-0.01}"
export IPT_ALL_HALF_TOL="${IPT_ALL_HALF_TOL:-1e-3}"
export IPT_ALL_HALF_MAXITER="${IPT_ALL_HALF_MAXITER:-1000}"
export IPT_FP64_TOL="${IPT_FP64_TOL:-1e-12}"
export IPT_FP64_MAXITER="${IPT_FP64_MAXITER:-1000}"
export IPT_ALL_HALF_WARMUP_N="${IPT_ALL_HALF_WARMUP_N:-128}"
export IPT_SEED="${IPT_SEED:-20260603}"

BIN="$TEST_DIR/benchmark_fp16_all_half_ipt_cuda"
SRC="$TEST_DIR/benchmark_fp16_all_half_ipt.cu"

echo "===== FP16 all-half IPT job ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "IPT_ALL_HALF_N=$IPT_ALL_HALF_N"
echo "IPT_ALL_HALF_EPSILON=$IPT_ALL_HALF_EPSILON"
echo "IPT_ALL_HALF_TOL=$IPT_ALL_HALF_TOL"
echo "IPT_ALL_HALF_MAXITER=$IPT_ALL_HALF_MAXITER"
echo "IPT_FP64_TOL=$IPT_FP64_TOL"
echo "IPT_FP64_MAXITER=$IPT_FP64_MAXITER"
echo "IPT_ALL_HALF_WARMUP_N=$IPT_ALL_HALF_WARMUP_N"
echo "IPT_ALL_HALF_CSV=${IPT_ALL_HALF_CSV:-}"
module list 2>&1
echo
nvidia-smi || true
echo

echo "===== build ====="
nvcc -O3 -std=c++14 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    "$SRC" -lcublas -o "$BIN"

echo
echo "===== run ====="
"$BIN"
status=$?

echo
echo "===== output files ====="
ls -lh "$LOG_DIR" "$RESULTS_DIR" || true
echo "finished at: $(date)"

exit "$status"
