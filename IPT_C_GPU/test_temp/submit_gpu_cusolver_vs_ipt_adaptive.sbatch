#!/usr/bin/env bash
#SBATCH --job-name=ipt_cusolver_cmp
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/adptive_logs/gpu_cusolver_vs_ipt_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/adptive_logs/gpu_cusolver_vs_ipt_%j.err

set -uo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
LOG_DIR="${IPT_C_LOG_DIR:-$ROOT/logs/adptive_logs}"
RESULTS_DIR="${IPT_C_RESULTS_DIR:-$ROOT/results}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

RUN_LOG="$LOG_DIR/gpu_cusolver_vs_ipt_${SLURM_JOB_ID:-manual}.log"
ln -sfn "$RUN_LOG" "$LOG_DIR/latest_gpu_cusolver_vs_ipt.log"
exec > >(tee "$RUN_LOG") 2>&1

if [ -f /fs1/software/modules/4.2.1-gcc8.4.1/init/bash ]; then
    source /fs1/software/modules/4.2.1-gcc8.4.1/init/bash
elif [ -f /etc/profile.d/module.sh ]; then
    source /etc/profile.d/module.sh
fi

module load GCC/12.2.0
module load CUDA/12.2

cd "$ROOT"

export IPT_C_ROOT="$ROOT"
export IPT_C_LOG_DIR="$LOG_DIR"
export IPT_C_RESULTS_DIR="$RESULTS_DIR"
export IPT_RESULT_CSV="${IPT_RESULT_CSV:-$RESULTS_DIR/gpu_cusolver_vs_ipt_results.csv}"
export IPT_SUMMARY_TXT="${IPT_SUMMARY_TXT:-$RESULTS_DIR/gpu_cusolver_vs_ipt_summary.txt}"
export IPT_BENCH_NS="${IPT_BENCH_NS:-32,64,128,256,512,1024,2048,4096,8192,16384,32768}"
export IPT_EPSILON="${IPT_EPSILON:-0.01}"
export IPT_TOL="${IPT_TOL:-1e-12}"
export IPT_MAXITER="${IPT_MAXITER:-1000}"
export IPT_REPEATS="${IPT_REPEATS:-10}"
export IPT_SEED="${IPT_SEED:-20260603}"
export IPT_COMPUTE_RESIDUALS="${IPT_COMPUTE_RESIDUALS:-true}"

echo "===== IPT_C GPU benchmark job ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "IPT_C_ROOT=$IPT_C_ROOT"
echo "IPT_C_RESULTS_DIR=$IPT_C_RESULTS_DIR"
echo "IPT_C_LOG_DIR=$IPT_C_LOG_DIR"
echo "IPT_RESULT_CSV=$IPT_RESULT_CSV"
echo "IPT_SUMMARY_TXT=$IPT_SUMMARY_TXT"
echo "IPT_BENCH_NS=$IPT_BENCH_NS"
echo "IPT_EPSILON=$IPT_EPSILON"
echo "IPT_TOL=$IPT_TOL"
echo "IPT_MAXITER=$IPT_MAXITER"
echo "IPT_REPEATS=$IPT_REPEATS"
echo "IPT_SEED=$IPT_SEED"
echo "IPT_COMPUTE_RESIDUALS=$IPT_COMPUTE_RESIDUALS"
module list 2>&1
echo

nvidia-smi || true
echo

./test/benchmark_gpu_syevd_vs_ipt_cuda
status=$?

echo
echo "===== output files ====="
ls -lh "$LOG_DIR" "$RESULTS_DIR" || true
echo "finished at: $(date)"

exit "$status"
