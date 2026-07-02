#!/usr/bin/env bash
#SBATCH --job-name=h2o_cas10
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/h2o_631g_cas10e10o_fci_63504_primme_vs_ipt_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/h2o_631g_cas10e10o_fci_63504_primme_vs_ipt_%j.err

set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
export IPT_C_ROOT="$ROOT"
export IPT_C_LOG_DIR="${IPT_C_LOG_DIR:-$ROOT/logs/sparse}"
export IPT_C_RESULTS_DIR="${IPT_C_RESULTS_DIR:-$ROOT/results/sparse}"

mkdir -p "$IPT_C_LOG_DIR" "$IPT_C_RESULTS_DIR"

bash "$ROOT/test/sparse/H2O_FCI/run_h2o_631g_cas10e10o_primme_vs_ipt.sh"
