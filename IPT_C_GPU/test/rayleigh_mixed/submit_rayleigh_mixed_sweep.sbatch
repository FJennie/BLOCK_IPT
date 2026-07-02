#!/usr/bin/env bash
#SBATCH --job-name=ipt_mixed_cmp
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/rayleigh_mixed/ipt_mixed_tf32_rayleigh_sweep_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/rayleigh_mixed/ipt_mixed_tf32_rayleigh_sweep_%j.err

set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
SRC_DIR="$ROOT/src/mixed_precision"
TEST_DIR="$ROOT/test/rayleigh_mixed"
LOG_DIR="${IPT_C_LOG_DIR:-$ROOT/logs/rayleigh_mixed}"
RESULTS_DIR="${IPT_C_RESULTS_DIR:-$ROOT/results/rayleigh_mixed}"

mkdir -p "$SRC_DIR" "$TEST_DIR" "$LOG_DIR" "$RESULTS_DIR"

RUN_LOG="$LOG_DIR/ipt_mixed_tf32_rayleigh_sweep_${SLURM_JOB_ID:-manual}.log"
ln -sfn "$RUN_LOG" "$LOG_DIR/latest_ipt_mixed_tf32_rayleigh_sweep.log"
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
export IPT_SUMMARY_TXT="${IPT_SUMMARY_TXT:-$RESULTS_DIR/ipt_mixed_tf32_rayleigh_sweep_summary.txt}"
export IPT_SWEEP_NS="${IPT_SWEEP_NS:-4096,8192,16384}"
export IPT_EPSILON_VALUES="${IPT_EPSILON_VALUES:-0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.12}"
export IPT_TF32_TOL="${IPT_TF32_TOL:-1e-6}"
export IPT_TF32_MAXITER="${IPT_TF32_MAXITER:-1000}"
export IPT_TOL="${IPT_TOL:-1e-12}"
export IPT_MAXITER="${IPT_MAXITER:-1000}"
export IPT_REPEATS="${IPT_REPEATS:-5}"
export IPT_WARMUP_N="${IPT_WARMUP_N:-128}"
export IPT_SEED="${IPT_SEED:-20260617}"
export IPT_APPEND_RESULTS="${IPT_APPEND_RESULTS:-false}"
export IPT_SKIP_EXISTING="${IPT_SKIP_EXISTING:-false}"

BIN="$TEST_DIR/benchmark_rayleigh_mixed_sweep_cuda"
SRC="$TEST_DIR/benchmark_rayleigh_mixed_sweep.cu"

echo "===== IPT mixed TF32 and Rayleigh mixed sweep job ====="
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
echo "IPT_SUMMARY_TXT=$IPT_SUMMARY_TXT"
echo "IPT_SWEEP_NS=$IPT_SWEEP_NS"
echo "IPT_EPSILON_VALUES=$IPT_EPSILON_VALUES"
echo "IPT_TF32_TOL=$IPT_TF32_TOL"
echo "IPT_TF32_MAXITER=$IPT_TF32_MAXITER"
echo "IPT_TOL=$IPT_TOL"
echo "IPT_MAXITER=$IPT_MAXITER"
echo "IPT_REPEATS=$IPT_REPEATS"
echo "IPT_WARMUP_N=$IPT_WARMUP_N"
echo "IPT_APPEND_RESULTS=$IPT_APPEND_RESULTS"
echo "IPT_SKIP_EXISTING=$IPT_SKIP_EXISTING"
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
