#!/usr/bin/env bash
#SBATCH --job-name=ch4_res_support
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/residual_support_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/CH4_K30_dubug/residual_support_%j.err

set -u -o pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
RESULT_DIR="${IPT_C_RESULTS_DIR:-$ROOT/results/sparse/CH4_K30_debug/jd_local_residual_support_v1}"
LOG_DIR="${IPT_C_LOG_DIR:-$ROOT/logs/sparse/CH4_K30_dubug/jd_local_residual_support_v1}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"
export IPT_C_RESULTS_DIR="$RESULT_DIR"
export IPT_C_LOG_DIR="$LOG_DIR"
export IPT_CH4_MATRIX_CACHE="${IPT_CH4_MATRIX_CACHE:-$ROOT/results/sparse/CH4_K30_debug/ch4_sto6g_fci_15876_csc.bin}"
export IPT_LOAD_MATRIX=1
export IPT_SAVE_MATRIX=0
export IPT_K=30
export IPT_REPEATS=1
export IPT_MAXITER=200
export IPT_BLOCK_CLUSTER_MAXITER=200
export IPT_BLOCK_CLUSTER_TOL=0
export IPT_BLOCK_CLUSTER=1
export IPT_BLOCK_CLUSTER_OVERSAMPLE=3
export IPT_BLOCK_CLUSTER_QR=1
export IPT_BLOCK_CLUSTER_ADAPTIVE=0
export IPT_DAVIDSON_ENRICH=1
export IPT_DAVIDSON_STEPS=80
export IPT_DAVIDSON_ACTIVE_MAX=2
export IPT_DAVIDSON_SELECT_TOL=1e-12
export IPT_DAVIDSON_PROTECT_TOL=1e-10
export IPT_DAVIDSON_LOCKED_TOL=1e-12
export IPT_DAVIDSON_DENOM_CLIP=3e-9
export IPT_DAVIDSON_ACCEPT_ONLY_IF_IMPROVES=1
export IPT_DAVIDSON_ORTHO_REPEATS=2
export IPT_DAVIDSON_RESTART_EVERY=20
export IPT_DAVIDSON_RESTART_KEEP_EXTRA=5
export IPT_DUMP_RESIDUAL_SUPPORT=1
export IPT_RESIDUAL_SUPPORT_PAIR=29
export IPT_RESIDUAL_SUPPORT_TOP=100
export IPT_JD_LOCAL_CORRECTION=0
export RUN_PRIMME=0
export RUN_WARMUP=0

bash "$ROOT/test/sparse/CH4/run_ch4_sto6g_fci_15876_ipt.sh"
