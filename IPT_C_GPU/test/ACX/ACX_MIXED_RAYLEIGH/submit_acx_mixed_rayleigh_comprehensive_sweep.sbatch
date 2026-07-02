#!/usr/bin/env bash
#SBATCH --job-name=acx_rayleigh_cmp
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/ACX/ACX_MIXED_RAYLEIGH/acx_mixed_rayleigh_comprehensive_sweep_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/ACX/ACX_MIXED_RAYLEIGH/acx_mixed_rayleigh_comprehensive_sweep_%j.err

set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
TEST_DIR="$ROOT/test/ACX/ACX_MIXED_RAYLEIGH"
LOG_DIR="${ACX_MIXED_RAYLEIGH_LOG_DIR:-$ROOT/logs/ACX/ACX_MIXED_RAYLEIGH}"
RESULTS_DIR="${ACX_MIXED_RAYLEIGH_RESULTS_DIR:-$ROOT/results/ACX/ACX_MIXED_RAYLEIGH}"

mkdir -p "$TEST_DIR" "$LOG_DIR" "$RESULTS_DIR"

RUN_LOG="$LOG_DIR/acx_mixed_rayleigh_comprehensive_sweep_${SLURM_JOB_ID:-manual}.log"
ln -sfn "$RUN_LOG" "$LOG_DIR/latest_acx_mixed_rayleigh_comprehensive_sweep.log"
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
export ACX_MIXED_RAYLEIGH_LOG_DIR="$LOG_DIR"
export ACX_MIXED_RAYLEIGH_RESULTS_DIR="$RESULTS_DIR"
export ACX_MIXED_RAYLEIGH_COMPREHENSIVE_CSV="${ACX_MIXED_RAYLEIGH_COMPREHENSIVE_CSV:-$RESULTS_DIR/acx_mixed_rayleigh_comprehensive_sweep.csv}"
export ACX_MIXED_RAYLEIGH_COMPREHENSIVE_SUMMARY_TXT="${ACX_MIXED_RAYLEIGH_COMPREHENSIVE_SUMMARY_TXT:-$RESULTS_DIR/acx_mixed_rayleigh_comprehensive_sweep_summary.txt}"
export ACX_MIXED_RAYLEIGH_SWEEP_NS="${ACX_MIXED_RAYLEIGH_SWEEP_NS:-4096,8192,16384}"
export ACX_MIXED_RAYLEIGH_EPSILON_VALUES="${ACX_MIXED_RAYLEIGH_EPSILON_VALUES:-0.06,0.08,0.1,0.12}"
export ACX_TF32_TOL="${ACX_TF32_TOL:-1e-6}"
export ACX_TF32_MAXITER="${ACX_TF32_MAXITER:-1000}"
export ACX_FP64_TOL="${ACX_FP64_TOL:-1e-12}"
export ACX_FP64_MAXITER="${ACX_FP64_MAXITER:-1000}"
export ACX_MIXED_RAYLEIGH_REPEATS="${ACX_MIXED_RAYLEIGH_REPEATS:-5}"
export ACX_MIXED_RAYLEIGH_WARMUP_N="${ACX_MIXED_RAYLEIGH_WARMUP_N:-128}"
export ACX_MIXED_RAYLEIGH_SEED="${ACX_MIXED_RAYLEIGH_SEED:-20260617}"
export ACX_MIXED_RAYLEIGH_APPEND_RESULTS="${ACX_MIXED_RAYLEIGH_APPEND_RESULTS:-false}"
export ACX_MIXED_RAYLEIGH_COMPUTE_RESIDUAL="${ACX_MIXED_RAYLEIGH_COMPUTE_RESIDUAL:-true}"

BIN="$TEST_DIR/benchmark_acx_mixed_rayleigh_comprehensive_sweep"
SRC="$TEST_DIR/benchmark_acx_mixed_rayleigh_comprehensive_sweep.cu"

echo "===== ACX mixed Rayleigh comprehensive sweep job ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "IPT_C_ROOT=$IPT_C_ROOT"
echo "TEST_DIR=$TEST_DIR"
echo "ACX_MIXED_RAYLEIGH_RESULTS_DIR=$ACX_MIXED_RAYLEIGH_RESULTS_DIR"
echo "ACX_MIXED_RAYLEIGH_LOG_DIR=$ACX_MIXED_RAYLEIGH_LOG_DIR"
echo "ACX_MIXED_RAYLEIGH_COMPREHENSIVE_CSV=$ACX_MIXED_RAYLEIGH_COMPREHENSIVE_CSV"
echo "ACX_MIXED_RAYLEIGH_COMPREHENSIVE_SUMMARY_TXT=$ACX_MIXED_RAYLEIGH_COMPREHENSIVE_SUMMARY_TXT"
echo "ACX_MIXED_RAYLEIGH_SWEEP_NS=$ACX_MIXED_RAYLEIGH_SWEEP_NS"
echo "ACX_MIXED_RAYLEIGH_EPSILON_VALUES=$ACX_MIXED_RAYLEIGH_EPSILON_VALUES"
echo "ACX_TF32_TOL=$ACX_TF32_TOL"
echo "ACX_TF32_MAXITER=$ACX_TF32_MAXITER"
echo "ACX_FP64_TOL=$ACX_FP64_TOL"
echo "ACX_FP64_MAXITER=$ACX_FP64_MAXITER"
echo "ACX_MIXED_RAYLEIGH_REPEATS=$ACX_MIXED_RAYLEIGH_REPEATS"
echo "ACX_MIXED_RAYLEIGH_WARMUP_N=$ACX_MIXED_RAYLEIGH_WARMUP_N"
echo "ACX_MIXED_RAYLEIGH_SEED=$ACX_MIXED_RAYLEIGH_SEED"
echo "ACX_MIXED_RAYLEIGH_APPEND_RESULTS=$ACX_MIXED_RAYLEIGH_APPEND_RESULTS"
echo "ACX_MIXED_RAYLEIGH_COMPUTE_RESIDUAL=$ACX_MIXED_RAYLEIGH_COMPUTE_RESIDUAL"
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
