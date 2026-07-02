#!/usr/bin/env bash
#SBATCH --job-name=ipt_c_tests
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C/logs/runtests_cuda_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C/logs/runtests_cuda_%j.err

set -uo pipefail

ROOT=/fs1/home/nudt_liujie/ftt/IPT_C
LOG_DIR="$ROOT/logs"
RESULTS_DIR="$ROOT/results"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

RUN_LOG="$LOG_DIR/run_runtests_cuda_${SLURM_JOB_ID:-manual}.log"
ln -sfn "$RUN_LOG" "$LOG_DIR/latest_run_runtests_cuda.log"
exec > >(tee "$RUN_LOG") 2>&1

if [ -f /fs1/software/modules/4.2.1-gcc8.4.1/init/bash ]; then
    source /fs1/software/modules/4.2.1-gcc8.4.1/init/bash
elif [ -f /etc/profile.d/module.sh ]; then
    source /etc/profile.d/module.sh
fi

module load GCC/12.2.0
module load CUDA/12.2

cd "$ROOT"

echo "===== IPT_C CUDA tests ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
module list 2>&1
echo

nvidia-smi || true
echo

IPT_C_RESULTS_DIR="$RESULTS_DIR" ./test/runtests_cuda
status=$?

echo
echo "===== output files ====="
ls -lh "$LOG_DIR" "$RESULTS_DIR" || true
echo "finished at: $(date)"

exit "$status"
