#!/usr/bin/env bash
#SBATCH --job-name=ipt_sparse
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/sparse_tests_%j.out
#SBATCH --error=/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/sparse_tests_%j.err

set -uo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
export IPT_C_ROOT="$ROOT"
export IPT_C_LOG_DIR="${IPT_C_LOG_DIR:-$ROOT/logs/sparse}"
export IPT_C_RESULTS_DIR="${IPT_C_RESULTS_DIR:-$ROOT/results/sparse}"

mkdir -p "$IPT_C_LOG_DIR" "$IPT_C_RESULTS_DIR"

bash "$ROOT/test/sparse/run_sparse_tests.sh"
