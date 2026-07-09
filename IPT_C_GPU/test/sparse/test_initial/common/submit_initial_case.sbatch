#!/usr/bin/env bash
#SBATCH --job-name=ti_case
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
CASE_ENV="${1:?missing case env}"
K_VALUE="${2:?missing K}"

bash "$ROOT/test/sparse/test_initial/common/run_initial_case.sh" \
    "$CASE_ENV" "$K_VALUE"
