#!/usr/bin/env bash
#SBATCH --job-name=acx_cuda_sweep
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/ACX/acx_cuda_sweep_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/ACX/acx_cuda_sweep_%j.err

set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
SRC_DIR="$ROOT/src/ACX"
TEST_DIR="$ROOT/test/ACX"
LOG_DIR="${ACX_LOG_DIR:-$ROOT/logs/ACX}"
RESULTS_DIR="${ACX_RESULTS_DIR:-$ROOT/results/ACX}"

mkdir -p "$SRC_DIR" "$TEST_DIR" "$LOG_DIR" "$RESULTS_DIR"

RUN_LOG="$LOG_DIR/acx_cuda_sweep_${SLURM_JOB_ID:-manual}.log"
ln -sfn "$RUN_LOG" "$LOG_DIR/latest_acx_cuda_sweep.log"
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
export ACX_LOG_DIR="$LOG_DIR"
export ACX_RESULTS_DIR="$RESULTS_DIR"
export ACX_SUMMARY_TXT="${ACX_SUMMARY_TXT:-$RESULTS_DIR/acx_cuda_sweep_summary.txt}"
export ACX_SWEEP_NS="${ACX_SWEEP_NS:-4096,8192,16384}"
export ACX_EPSILON_VALUES="${ACX_EPSILON_VALUES:-0.06,0.08,0.1,0.12}"
export ACX_TOL="${ACX_TOL:-1e-12}"
export ACX_MAXITER="${ACX_MAXITER:-1000}"
export ACX_REPEATS="${ACX_REPEATS:-5}"
export ACX_WARMUP_N="${ACX_WARMUP_N:-128}"
export ACX_SEED="${ACX_SEED:-20260615}"
export ACX_APPEND_RESULTS="${ACX_APPEND_RESULTS:-false}"
export ACX_COMPUTE_RESIDUAL="${ACX_COMPUTE_RESIDUAL:-true}"

BIN="$TEST_DIR/benchmark_acx_cuda_sweep"
SRC="$TEST_DIR/benchmark_acx_cuda_sweep.cu"

echo "===== ACX CUDA sweep job ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "IPT_C_ROOT=$IPT_C_ROOT"
echo "SRC_DIR=$SRC_DIR"
echo "TEST_DIR=$TEST_DIR"
echo "ACX_RESULTS_DIR=$ACX_RESULTS_DIR"
echo "ACX_LOG_DIR=$ACX_LOG_DIR"
echo "ACX_SUMMARY_TXT=$ACX_SUMMARY_TXT"
echo "ACX_SWEEP_NS=$ACX_SWEEP_NS"
echo "ACX_EPSILON_VALUES=$ACX_EPSILON_VALUES"
echo "ACX_TOL=$ACX_TOL"
echo "ACX_MAXITER=$ACX_MAXITER"
echo "ACX_REPEATS=$ACX_REPEATS"
echo "ACX_WARMUP_N=$ACX_WARMUP_N"
echo "ACX_APPEND_RESULTS=$ACX_APPEND_RESULTS"
echo "ACX_COMPUTE_RESIDUAL=$ACX_COMPUTE_RESIDUAL"
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

echo
echo "===== output files ====="
ls -lh "$LOG_DIR" "$RESULTS_DIR" || true
echo "finished at: $(date)"
