#!/usr/bin/env bash
#SBATCH --job-name=rayleigh_vs_cusolver
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/rayleigh_mixed/rayleigh_vs_cusolver_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/rayleigh_mixed/rayleigh_vs_cusolver_%j.err

set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
RAY_TEST_DIR="$ROOT/test/rayleigh_mixed"
TF32_TEST_DIR="$ROOT/test/mixed_precision_tf32"
LOG_DIR="${IPT_C_LOG_DIR:-$ROOT/logs/rayleigh_mixed}"
RAY_RESULTS_DIR="${IPT_RAYLEIGH_RESULTS_DIR:-$ROOT/results/rayleigh_mixed}"
TF32_RESULTS_DIR="${IPT_TF32_RESULTS_DIR:-$ROOT/results/mixed_precision_tf32}"

mkdir -p "$LOG_DIR" "$RAY_RESULTS_DIR" "$TF32_RESULTS_DIR"

RUN_LOG="$LOG_DIR/rayleigh_vs_cusolver_${SLURM_JOB_ID:-manual}.log"
ln -sfn "$RUN_LOG" "$LOG_DIR/latest_rayleigh_vs_cusolver.log"
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

NS="${IPT_SWEEP_NS:-8192,16384,26000}"
EPS="${IPT_EPSILON_VALUES:-0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50}"
REPEATS="${IPT_REPEATS:-5}"
WARMUP_N="${IPT_WARMUP_N:-128}"
TF32_TOL="${IPT_TF32_TOL:-1e-6}"
FP64_TOL="${IPT_TOL:-1e-12}"
MAXITER="${IPT_MAXITER:-1000}"
TF32_MAXITER="${IPT_TF32_MAXITER:-1000}"
SEED="${IPT_SEED:-20260614}"

RAY_BIN="$RAY_TEST_DIR/benchmark_rayleigh_mixed_sweep_cuda"
RAY_SRC="$RAY_TEST_DIR/benchmark_rayleigh_mixed_sweep.cu"
TF32_BIN="$TF32_TEST_DIR/benchmark_tf32_cusolver_sweep_cuda"
TF32_SRC="$TF32_TEST_DIR/benchmark_tf32_cusolver_sweep.cu"

DETAIL_CSV="$RAY_RESULTS_DIR/rayleigh_vs_cusolver_detail_${SLURM_JOB_ID:-manual}.csv"
THRESHOLD_CSV="$RAY_RESULTS_DIR/rayleigh_vs_cusolver_threshold_${SLURM_JOB_ID:-manual}.csv"
MISSING_EPS_FILE="$RAY_RESULTS_DIR/rayleigh_vs_cusolver_missing_threshold_eps_${SLURM_JOB_ID:-manual}.txt"

echo "===== Rayleigh mixed QR vs cuSolver threshold job ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "ROOT=$ROOT"
echo "NS=$NS"
echo "EPS=$EPS"
echo "REPEATS=$REPEATS"
echo "WARMUP_N=$WARMUP_N"
echo "TF32_TOL=$TF32_TOL"
echo "FP64_TOL=$FP64_TOL"
echo "MAXITER=$MAXITER"
echo "TF32_MAXITER=$TF32_MAXITER"
echo "RAY_RESULTS_DIR=$RAY_RESULTS_DIR"
echo "TF32_RESULTS_DIR=$TF32_RESULTS_DIR"
module list 2>&1
echo
nvidia-smi || true
echo

echo "===== build rayleigh sweep ====="
nvcc -O3 -std=c++14 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    "$RAY_SRC" -lcublas -lcusolver -o "$RAY_BIN"

echo "===== build tf32/cusolver sweep ====="
nvcc -O3 -std=c++14 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    "$TF32_SRC" -lcublas -lcusolver -o "$TF32_BIN"

echo
echo "===== run Rayleigh mixed QR sweep ====="
export IPT_C_ROOT="$ROOT"
export IPT_C_RESULTS_DIR="$RAY_RESULTS_DIR"
export IPT_C_LOG_DIR="$LOG_DIR"
export IPT_SUMMARY_TXT="$RAY_RESULTS_DIR/rayleigh_mixed_qr_threshold_sweep_${SLURM_JOB_ID:-manual}.txt"
export IPT_SWEEP_NS="$NS"
export IPT_EPSILON_VALUES="$EPS"
export IPT_TF32_TOL="$TF32_TOL"
export IPT_TF32_MAXITER="$TF32_MAXITER"
export IPT_TOL="$FP64_TOL"
export IPT_MAXITER="$MAXITER"
export IPT_REPEATS="$REPEATS"
export IPT_WARMUP_N="$WARMUP_N"
export IPT_SEED="$SEED"
export IPT_APPEND_RESULTS=false
export IPT_SKIP_EXISTING=false
"$RAY_BIN"

echo
echo "===== estimate first crossing and collect exact threshold epsilon list ====="
export IPT_COMPARE_NS="$NS"
export IPT_COMPARE_EPSILONS="$EPS"
export IPT_RAYLEIGH_RESULTS_DIR="$RAY_RESULTS_DIR"
export IPT_TF32_RESULTS_DIR="$TF32_RESULTS_DIR"
export IPT_COMPARE_DETAIL_CSV="$DETAIL_CSV"
export IPT_COMPARE_THRESHOLD_CSV="$THRESHOLD_CSV"
export IPT_MISSING_THRESHOLD_EPS_FILE="$MISSING_EPS_FILE"
python3 "$RAY_TEST_DIR/summarize_rayleigh_vs_cusolver.py"

THRESHOLD_EPS=""
if [ -s "$MISSING_EPS_FILE" ]; then
    THRESHOLD_EPS="$(tr -d '[:space:]' < "$MISSING_EPS_FILE")"
fi

if [ -n "$THRESHOLD_EPS" ]; then
    echo
    echo "===== run exact TF32 mixed/original IPT/cuSolver for threshold epsilon(s): $THRESHOLD_EPS ====="
    export IPT_C_RESULTS_DIR="$TF32_RESULTS_DIR"
    export IPT_C_LOG_DIR="$ROOT/logs/mixed_precision_tf32"
    mkdir -p "$IPT_C_LOG_DIR"
    export IPT_SUMMARY_TXT="$TF32_RESULTS_DIR/tf32_cusolver_threshold_exact_${SLURM_JOB_ID:-manual}.txt"
    export IPT_SWEEP_NS="$NS"
    export IPT_EPSILON_VALUES="$THRESHOLD_EPS"
    export IPT_FP16_TOL="${IPT_FP16_TOL:-1e-3}"
    export IPT_FP16_MAXITER="${IPT_FP16_MAXITER:-1000}"
    export IPT_TF32_TOL="$TF32_TOL"
    export IPT_TF32_MAXITER="$TF32_MAXITER"
    export IPT_TOL="$FP64_TOL"
    export IPT_MAXITER="$MAXITER"
    export IPT_REPEATS="$REPEATS"
    export IPT_WARMUP_N="$WARMUP_N"
    export IPT_SEED="$SEED"
    export IPT_COMPUTE_RESIDUALS=true
    export IPT_APPEND_RESULTS=false
    export IPT_SKIP_EXISTING=false
    export IPT_RUN_FP16=false
    export IPT_RUN_TF32=true
    "$TF32_BIN"

    echo
    echo "===== final summary with exact threshold baseline ====="
    export IPT_COMPARE_NS="$NS"
    export IPT_COMPARE_EPSILONS="$EPS"
    export IPT_RAYLEIGH_RESULTS_DIR="$RAY_RESULTS_DIR"
    export IPT_TF32_RESULTS_DIR="$TF32_RESULTS_DIR"
    export IPT_COMPARE_DETAIL_CSV="$DETAIL_CSV"
    export IPT_COMPARE_THRESHOLD_CSV="$THRESHOLD_CSV"
    export IPT_MISSING_THRESHOLD_EPS_FILE="$MISSING_EPS_FILE"
    python3 "$RAY_TEST_DIR/summarize_rayleigh_vs_cusolver.py"
else
    echo
    echo "No Rayleigh slower-than-cuSolver crossing found in EPS range."
fi

echo
echo "===== threshold summary ====="
cat "$THRESHOLD_CSV"
echo
echo "===== detail summary path ====="
echo "$DETAIL_CSV"
echo "finished at: $(date)"
