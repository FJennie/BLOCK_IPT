#!/usr/bin/env bash
set -euo pipefail
ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
PRIMME_ROOT="${PRIMME_ROOT:-/fs1/home/nudt_liujie/ftt/primme-3.2.3}"
PYTHON_ROOT="${PYTHON_ROOT:-/fs1/software/python/3.8_miniconda_4.10.3}"
PYSCF_SITE_PACKAGES="${PYSCF_SITE_PACKAGES:-/fs1/software/hpcsystem/THL7/software/pyscf/2.0.1-py3.8/lib/python3.8/site-packages}"
LOG_DIR="${IPT_C_LOG_DIR:-$ROOT/logs/sparse}"
RESULTS_DIR="${IPT_C_RESULTS_DIR:-$ROOT/results/sparse}"
TEST_DIR="$ROOT/test/sparse/CO"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$TEST_DIR"
RUN_LOG="$LOG_DIR/co_sto6g_fci_14400_ipt_${SLURM_JOB_ID:-manual}.log"
ln -sfn "$RUN_LOG" "$LOG_DIR/latest_co_sto6g_fci_14400_ipt.log"
exec > >(tee "$RUN_LOG") 2>&1
if [ -f /fs1/software/modules/4.2.1-gcc8.4.1/init/bash ]; then source /fs1/software/modules/4.2.1-gcc8.4.1/init/bash; elif [ -f /etc/profile.d/module.sh ]; then source /etc/profile.d/module.sh; fi
module unload CUDA/12.8.1 2>/dev/null || true
module unload CUDA/12.2 2>/dev/null || true
module load GCC/12.2.0
module load CUDA/12.2
cd "$ROOT"
export IPT_C_ROOT="$ROOT"
export IPT_C_LOG_DIR="$LOG_DIR"
export IPT_C_RESULTS_DIR="$RESULTS_DIR"
export IPT_REPEATS="${IPT_REPEATS:-1}"
export IPT_TOL="${IPT_TOL:-1e-12}"
export IPT_MAXITER="${IPT_MAXITER:-1000}"
export PRIMME_MAX_MATVECS="${PRIMME_MAX_MATVECS:-1000}"
export RUN_PRIMME="${RUN_PRIMME:-0}"
export RUN_WARMUP="${RUN_WARMUP:-0}"
export IPT_EST_NNZ_PER_COL="${IPT_EST_NNZ_PER_COL:-900}"
export PYSCF_SITE_PACKAGES
export PYTHONPATH="$PYSCF_SITE_PACKAGES:${PYTHONPATH:-}"
OPENBLAS_LIB="${OPENBLAS_LIB:-/fs1/software/cp2k/8.1-gcc8.4-openmpi/lib/libopenblas.so.0}"
if [ ! -f "$OPENBLAS_LIB" ]; then OPENBLAS_LIB="/usr/lib64/libopenblas.so.0"; fi
if [ ! -f "$OPENBLAS_LIB" ]; then OPENBLAS_LIB="/lib64/libopenblas.so.0"; fi
OPENBLAS_DIR="$(dirname "$OPENBLAS_LIB")"
GCC_LIB_DIR="${GCC_LIB_DIR:-/fs1/software/gcc/12.2.0/lib64}"
ln -sfn "$OPENBLAS_LIB" "$TEST_DIR/libopenblas.so"
export LD_LIBRARY_PATH="$GCC_LIB_DIR:$PYTHON_ROOT/lib:$PYSCF_SITE_PACKAGES/numpy.libs:$PYSCF_SITE_PACKAGES/scipy.libs:$PYSCF_SITE_PACKAGES/h5py.libs:$PYSCF_SITE_PACKAGES/pyscf/lib:$OPENBLAS_DIR:$TEST_DIR:${LD_LIBRARY_PATH:-}"
echo "===== CO/STO-6G full FCI IPT sparse benchmark ====="
date; hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "IPT_REPEATS=$IPT_REPEATS"
echo "IPT_TOL=$IPT_TOL"
echo "IPT_MAXITER=$IPT_MAXITER"
echo "RUN_PRIMME=$RUN_PRIMME"
echo "RUN_WARMUP=$RUN_WARMUP"
echo "IPT_EST_NNZ_PER_COL=$IPT_EST_NNZ_PER_COL"
module list 2>&1
"$PYTHON_ROOT/bin/python3" - <<'PY'
import importlib
for name in ("numpy", "scipy", "pyscf"):
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "ok"), getattr(mod, "__file__", ""))
PY
nvcc --version
nvidia-smi || true
echo "===== build ====="
set -x
nvcc -O3 -std=c++14 -lineinfo -arch=sm_80     -I"$PRIMME_ROOT/include"     -I"$PYTHON_ROOT/include/python3.8"     "$TEST_DIR/benchmark_co_sto6g_fci_14400_ipt.cu"     "$PRIMME_ROOT/lib/libprimme.a"     -o "$TEST_DIR/benchmark_co_sto6g_fci_14400_ipt_cuda"     -lcublas -lcusparse -lcusolver -lcudart     -L"$PYTHON_ROOT/lib" -lpython3.8     -L"$TEST_DIR" -lopenblas     -lpthread -ldl -lutil -lrt     -Xlinker -rpath -Xlinker "$GCC_LIB_DIR"     -Xlinker -rpath -Xlinker "$PYTHON_ROOT/lib"     -Xlinker -rpath -Xlinker "$OPENBLAS_DIR"     -lgfortran "$GCC_LIB_DIR/libstdc++.so" -lm
set +x
echo "===== run ====="
"$TEST_DIR/benchmark_co_sto6g_fci_14400_ipt_cuda"
status=$?
echo "===== output files ====="
ls -lh "$LOG_DIR" "$RESULTS_DIR" || true
echo "finished at: $(date)"
exit "$status"
