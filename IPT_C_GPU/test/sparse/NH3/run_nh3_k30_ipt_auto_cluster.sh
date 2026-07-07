#!/usr/bin/env bash

set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
PRIMME_ROOT="${PRIMME_ROOT:-/fs1/home/nudt_liujie/ftt/primme-3.2.3}"
PYTHON_ROOT="${PYTHON_ROOT:-/fs1/software/python/3.8_miniconda_4.10.3}"
PYSCF_SITE_PACKAGES="${PYSCF_SITE_PACKAGES:-/fs1/software/hpcsystem/THL7/software/pyscf/2.0.1-py3.8/lib/python3.8/site-packages}"

TEST_DIR="$ROOT/test/sparse/NH3"
LOG_DIR="${IPT_C_LOG_DIR:-$ROOT/logs/sparse/NH3_K30/manual}"
RESULTS_DIR="${IPT_C_RESULTS_DIR:-$ROOT/results/sparse/NH3_k30/manual}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$TEST_DIR"

BUILD_DIR="${IPT_BENCH_BUILD_DIR:-$TEST_DIR/.codex_build}"
mkdir -p "$BUILD_DIR"

if [ "${IPT_SKIP_BUILD:-0}" = "1" ]; then
  BENCH_BIN="${IPT_BENCH_BIN:-$TEST_DIR/benchmark_nh3_sto6g_fci_3136_ipt_cuda}"
else
  BENCH_BIN="${IPT_BENCH_BIN:-$BUILD_DIR/benchmark_nh3_sto6g_fci_3136_ipt_cuda_${SLURM_JOB_ID:-manual}_$$}"
fi

RUN_LOG="$LOG_DIR/nh3_sto6g_fci_3136_ipt_k30_${SLURM_JOB_ID:-manual}.log"
ln -sfn "$RUN_LOG" "$LOG_DIR/latest_nh3_sto6g_fci_3136_ipt_k30.log"

exec > >(tee "$RUN_LOG") 2>&1

if [ -f /fs1/software/modules/4.2.1-gcc8.4.1/init/bash ]; then
  source /fs1/software/modules/4.2.1-gcc8.4.1/init/bash
elif [ -f /etc/profile.d/module.sh ]; then
  source /etc/profile.d/module.sh
fi

module unload CUDA/12.8.1 2>/dev/null || true
module unload CUDA/12.2 2>/dev/null || true
module load GCC/12.2.0
module load CUDA/12.2

cd "$ROOT"

export IPT_C_ROOT="$ROOT"
export IPT_C_LOG_DIR="$LOG_DIR"
export IPT_C_RESULTS_DIR="$RESULTS_DIR"

# NH3 前 30 个特征对
export IPT_K="${IPT_K:-30}"
export IPT_REPEATS="${IPT_REPEATS:-1}"
export IPT_TOL="${IPT_TOL:-1e-12}"
export IPT_MAXITER="${IPT_MAXITER:-200}"

# block-cluster / Davidson 求解参数；不指定任何手工 active pair 或手工 cluster。
export IPT_DEBUG_PREPARATION="${IPT_DEBUG_PREPARATION:-1}"
export IPT_DEGENERACY_THRESHOLD="${IPT_DEGENERACY_THRESHOLD:-0}"

export IPT_BLOCK_CLUSTER="${IPT_BLOCK_CLUSTER:-1}"
export IPT_BLOCK_CLUSTER_MAXITER="${IPT_BLOCK_CLUSTER_MAXITER:-200}"
export IPT_BLOCK_CLUSTER_TOL="${IPT_BLOCK_CLUSTER_TOL:-0}"
export IPT_BLOCK_CLUSTER_DAMPING="${IPT_BLOCK_CLUSTER_DAMPING:-1}"
export IPT_BLOCK_CLUSTER_OVERSAMPLE="${IPT_BLOCK_CLUSTER_OVERSAMPLE:-3}"
export IPT_BLOCK_CLUSTER_QR="${IPT_BLOCK_CLUSTER_QR:-1}"
export IPT_BLOCK_CLUSTER_ADAPTIVE="${IPT_BLOCK_CLUSTER_ADAPTIVE:-0}"

export IPT_DAVIDSON_ENRICH="${IPT_DAVIDSON_ENRICH:-1}"
export IPT_DAVIDSON_STEPS="${IPT_DAVIDSON_STEPS:-30}"
export IPT_DAVIDSON_EXTRA_STEPS="${IPT_DAVIDSON_EXTRA_STEPS:-20}"
export IPT_DAVIDSON_SELECT_TOL="${IPT_DAVIDSON_SELECT_TOL:-1e-12}"
export IPT_DAVIDSON_PROTECT_TOL="${IPT_DAVIDSON_PROTECT_TOL:-1e-10}"
export IPT_DAVIDSON_CONVERGED_TOL="${IPT_DAVIDSON_CONVERGED_TOL:-1e-13}"
export IPT_DAVIDSON_BLOCK_ACTIVE="${IPT_DAVIDSON_BLOCK_ACTIVE:-1}"
export IPT_DAVIDSON_DENOM_CLIP="${IPT_DAVIDSON_DENOM_CLIP:-3e-9}"
export IPT_DAVIDSON_ACCEPT_ONLY_IF_IMPROVES="${IPT_DAVIDSON_ACCEPT_ONLY_IF_IMPROVES:-1}"
export IPT_DAVIDSON_USE_BEST_SO_FAR="${IPT_DAVIDSON_USE_BEST_SO_FAR:-1}"
export IPT_DAVIDSON_RELAXED_ACCEPT="${IPT_DAVIDSON_RELAXED_ACCEPT:-1}"
export IPT_DAVIDSON_ACCEPT_REL_SLACK="${IPT_DAVIDSON_ACCEPT_REL_SLACK:-1e-12}"
export IPT_DAVIDSON_ACCEPT_ABS_SLACK="${IPT_DAVIDSON_ACCEPT_ABS_SLACK:-1e-15}"
export IPT_DAVIDSON_ACTIVE_PAIR_ACCEPT="${IPT_DAVIDSON_ACTIVE_PAIR_ACCEPT:-1}"
export IPT_DAVIDSON_LOCKED_DEGRADE_SLACK="${IPT_DAVIDSON_LOCKED_DEGRADE_SLACK:-1e-8}"
export IPT_DAVIDSON_RETRY_ON_REJECT="${IPT_DAVIDSON_RETRY_ON_REJECT:-1}"
export IPT_DAVIDSON_RETRY_DAMPING_LIST="${IPT_DAVIDSON_RETRY_DAMPING_LIST:-0.5,0.25}"
export IPT_DAVIDSON_RETRY_DENOM_CLIP_MULTS="${IPT_DAVIDSON_RETRY_DENOM_CLIP_MULTS:-10,100}"
export IPT_DAVIDSON_MIN_ACCEPTED_STEPS="${IPT_DAVIDSON_MIN_ACCEPTED_STEPS:-12}"
export IPT_DAVIDSON_ORTHO_REPEATS="${IPT_DAVIDSON_ORTHO_REPEATS:-2}"
export IPT_DAVIDSON_RESTART_EVERY="${IPT_DAVIDSON_RESTART_EVERY:-0}"
export IPT_DAVIDSON_RESTART_KEEP_EXTRA="${IPT_DAVIDSON_RESTART_KEEP_EXTRA:-5}"

export IPT_DUMP_PAIR_RESIDUALS="${IPT_DUMP_PAIR_RESIDUALS:-1}"
export IPT_DUMP_RESIDUAL_SUPPORT="${IPT_DUMP_RESIDUAL_SUPPORT:-0}"
export IPT_JD_LOCAL_CORRECTION="${IPT_JD_LOCAL_CORRECTION:-0}"

export RUN_PRIMME="${RUN_PRIMME:-0}"
export RUN_WARMUP="${RUN_WARMUP:-0}"
export IPT_EST_NNZ_PER_COL="${IPT_EST_NNZ_PER_COL:-600}"

# 再次清掉旧的手工 active/cluster 变量。
unset IPT_COMPARE_FORCE_ACTIVE_PAIRS
unset IPT_DAVIDSON_FORCE_ACTIVE_PAIRS
unset IPT_DAVIDSON_ACTIVE_CLUSTERS
unset IPT_DAVIDSON_ACTIVE_TOL
unset IPT_DAVIDSON_ACTIVE_MAX
unset IPT_DAVIDSON_CORRECTION_MAX_PER_STEP
unset IPT_DAVIDSON_DEBUG_ACTIVE_MAX

export PYSCF_SITE_PACKAGES
export PYTHONPATH="$PYSCF_SITE_PACKAGES:${PYTHONPATH:-}"

OPENBLAS_LIB="${OPENBLAS_LIB:-/fs1/software/cp2k/8.1-gcc8.4-openmpi/lib/libopenblas.so.0}"
if [ ! -f "$OPENBLAS_LIB" ]; then OPENBLAS_LIB="/usr/lib64/libopenblas.so.0"; fi
if [ ! -f "$OPENBLAS_LIB" ]; then OPENBLAS_LIB="/lib64/libopenblas.so.0"; fi
OPENBLAS_DIR="$(dirname "$OPENBLAS_LIB")"

GCC_LIB_DIR="${GCC_LIB_DIR:-/fs1/software/gcc/12.2.0/lib64}"

ln -sfn "$OPENBLAS_LIB" "$TEST_DIR/libopenblas.so"

export LD_LIBRARY_PATH="$GCC_LIB_DIR:$PYTHON_ROOT/lib:$PYSCF_SITE_PACKAGES/numpy.libs:$PYSCF_SITE_PACKAGES/scipy.libs:$PYSCF_SITE_PACKAGES/h5py.libs:$PYSCF_SITE_PACKAGES/pyscf/lib:$OPENBLAS_DIR:$TEST_DIR:${LD_LIBRARY_PATH:-}"

echo "===== NH3/STO-6G full FCI IPT sparse k=30 automatic cluster benchmark ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "BENCH_BIN=$BENCH_BIN"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "LOG_DIR=$LOG_DIR"

echo "IPT_K=$IPT_K"
echo "IPT_REPEATS=$IPT_REPEATS"
echo "IPT_TOL=$IPT_TOL"
echo "IPT_MAXITER=$IPT_MAXITER"

echo "IPT_BLOCK_CLUSTER=$IPT_BLOCK_CLUSTER"
echo "IPT_BLOCK_CLUSTER_MAXITER=$IPT_BLOCK_CLUSTER_MAXITER"
echo "IPT_BLOCK_CLUSTER_TOL=$IPT_BLOCK_CLUSTER_TOL"
echo "IPT_BLOCK_CLUSTER_OVERSAMPLE=$IPT_BLOCK_CLUSTER_OVERSAMPLE"
echo "IPT_BLOCK_CLUSTER_QR=$IPT_BLOCK_CLUSTER_QR"
echo "IPT_BLOCK_CLUSTER_ADAPTIVE=$IPT_BLOCK_CLUSTER_ADAPTIVE"

echo "IPT_DAVIDSON_ENRICH=$IPT_DAVIDSON_ENRICH"
echo "IPT_DAVIDSON_STEPS=$IPT_DAVIDSON_STEPS"
echo "IPT_DAVIDSON_EXTRA_STEPS=$IPT_DAVIDSON_EXTRA_STEPS"
echo "IPT_DAVIDSON_BLOCK_ACTIVE=$IPT_DAVIDSON_BLOCK_ACTIVE"
echo "IPT_DAVIDSON_DENOM_CLIP=$IPT_DAVIDSON_DENOM_CLIP"
echo "IPT_DAVIDSON_CONVERGED_TOL=$IPT_DAVIDSON_CONVERGED_TOL"
echo "IPT_DAVIDSON_MIN_ACCEPTED_STEPS=$IPT_DAVIDSON_MIN_ACCEPTED_STEPS"
echo "IPT_DAVIDSON_ORTHO_REPEATS=$IPT_DAVIDSON_ORTHO_REPEATS"
echo "IPT_DAVIDSON_RESTART_EVERY=$IPT_DAVIDSON_RESTART_EVERY"

echo "IPT_DAVIDSON_FORCE_ACTIVE_PAIRS=${IPT_DAVIDSON_FORCE_ACTIVE_PAIRS-<unset>}"
echo "IPT_DAVIDSON_ACTIVE_CLUSTERS=${IPT_DAVIDSON_ACTIVE_CLUSTERS-<unset>}"
echo "IPT_DAVIDSON_ACTIVE_TOL=${IPT_DAVIDSON_ACTIVE_TOL-<unset>}"
echo "IPT_DAVIDSON_ACTIVE_MAX=${IPT_DAVIDSON_ACTIVE_MAX-<unset>}"
echo "IPT_DAVIDSON_CORRECTION_MAX_PER_STEP=${IPT_DAVIDSON_CORRECTION_MAX_PER_STEP-<unset>}"

echo "RUN_PRIMME=$RUN_PRIMME"
echo "RUN_WARMUP=$RUN_WARMUP"
echo "IPT_EST_NNZ_PER_COL=$IPT_EST_NNZ_PER_COL"

module list 2>&1 || true

"$PYTHON_ROOT/bin/python3" - <<'PY'
import importlib
for name in ("numpy", "scipy", "pyscf"):
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "ok"), getattr(mod, "__file__", ""))
PY

nvcc --version
nvidia-smi || true

echo "===== build ====="
if [ "${IPT_SKIP_BUILD:-0}" = "1" ]; then
  echo "IPT_SKIP_BUILD=1; using existing benchmark binary"
else
  set -x
  nvcc -O3 -std=c++14 -lineinfo \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    -I"$PRIMME_ROOT/include" \
    -I"$PYTHON_ROOT/include/python3.8" \
    "$TEST_DIR/benchmark_nh3_sto6g_fci_3136_ipt.cu" \
    "$PRIMME_ROOT/lib/libprimme.a" \
    -o "$BENCH_BIN" \
    -lcublas -lcusparse -lcusolver -lcudart \
    -L"$PYTHON_ROOT/lib" -lpython3.8 \
    -L"$TEST_DIR" -lopenblas \
    -lpthread -ldl -lutil -lrt \
    -Xlinker -rpath -Xlinker "$GCC_LIB_DIR" \
    -Xlinker -rpath -Xlinker "$PYTHON_ROOT/lib" \
    -Xlinker -rpath -Xlinker "$OPENBLAS_DIR" \
    -lgfortran "$GCC_LIB_DIR/libstdc++.so" -lm
  set +x
fi

echo "===== run ====="
set +e
"$BENCH_BIN"
status=$?
set -e

echo "===== output files ====="
ls -lh "$LOG_DIR" "$RESULTS_DIR" || true

echo "finished at: $(date)"
exit "$status"