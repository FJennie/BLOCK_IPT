#!/usr/bin/env bash
#SBATCH --job-name=ch4_ipt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/ch4_sto6g_fci_15876_ipt_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/ch4_sto6g_fci_15876_ipt_%j.err
set -euo pipefail
ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
export IPT_C_ROOT="$ROOT"
export IPT_C_LOG_DIR="${IPT_C_LOG_DIR:-$ROOT/logs/sparse}"
export IPT_C_RESULTS_DIR="${IPT_C_RESULTS_DIR:-$ROOT/results/sparse}"
mkdir -p "$IPT_C_LOG_DIR" "$IPT_C_RESULTS_DIR"
bash "$ROOT/test/sparse/CH4/run_ch4_sto6g_fci_15876_ipt.sh"
