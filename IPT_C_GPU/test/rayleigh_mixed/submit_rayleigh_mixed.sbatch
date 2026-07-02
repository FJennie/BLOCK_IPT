#!/usr/bin/env bash
#SBATCH --job-name=rayleigh_mixed
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/rayleigh_mixed/rayleigh_mixed_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/rayleigh_mixed/rayleigh_mixed_%j.err

set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
SRC_DIR="$ROOT/src/mixed_precision"
TEST_DIR="$ROOT/test/rayleigh_mixed"
LOG_DIR="${IPT_C_LOG_DIR:-$ROOT/logs/rayleigh_mixed}"
RESULTS_DIR="${IPT_C_RESULTS_DIR:-$ROOT/results/rayleigh_mixed}"

mkdir -p "$SRC_DIR" "$TEST_DIR" "$LOG_DIR" "$RESULTS_DIR"

RUN_LOG="$LOG_DIR/rayleigh_mixed_${SLURM_JOB_ID:-manual}.log"
ln -sfn "$RUN_LOG" "$LOG_DIR/latest_rayleigh_mixed.log"
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
export IPT_N="${IPT_N:-16384}"
export IPT_EPSILON="${IPT_EPSILON:-0.07}"
export IPT_TF32_TOL="${IPT_TF32_TOL:-1e-6}"
export IPT_TF32_MAXITER="${IPT_TF32_MAXITER:-1000}"
export IPT_TOL="${IPT_TOL:-1e-12}"
export IPT_MAXITER="${IPT_MAXITER:-1000}"
export IPT_REPEATS="${IPT_REPEATS:-1}"
export IPT_WARMUP_N="${IPT_WARMUP_N:-128}"
export IPT_SEED="${IPT_SEED:-20260613}"
export IPT_COMPUTE_RESIDUALS="${IPT_COMPUTE_RESIDUALS:-true}"

BIN="$TEST_DIR/benchmark_rayleigh_mixed_cuda"
SRC="$TEST_DIR/benchmark_rayleigh_mixed.cu"

echo "===== Rayleigh mixed job ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "IPT_C_ROOT=$IPT_C_ROOT"
echo "SRC_DIR=$SRC_DIR"
echo "TEST_DIR=$TEST_DIR"
echo "IPT_C_RESULTS_DIR=$IPT_C_RESULTS_DIR"
echo "IPT_C_LOG_DIR=$IPT_C_LOG_DIR"
echo "IPT_N=$IPT_N"
echo "IPT_EPSILON=$IPT_EPSILON"
echo "IPT_TF32_TOL=$IPT_TF32_TOL"
echo "IPT_TF32_MAXITER=$IPT_TF32_MAXITER"
echo "IPT_TOL=$IPT_TOL"
echo "IPT_MAXITER=$IPT_MAXITER"
echo "IPT_REPEATS=$IPT_REPEATS"
echo "IPT_WARMUP_N=$IPT_WARMUP_N"
echo "IPT_COMPUTE_RESIDUALS=$IPT_COMPUTE_RESIDUALS"
module list 2>&1
echo
nvidia-smi || true
echo

echo "===== build ====="
nvcc -O3 -std=c++14 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    "$SRC" -lcublas -lcusolver -o "$BIN"

echo
echo "===== run ====="
"$BIN"

echo
echo "===== output files ====="
ls -lh "$LOG_DIR" "$RESULTS_DIR" || true
echo "finished at: $(date)"
