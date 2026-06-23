#!/bin/bash
# 在 WSL2 + JAX GPU 上跑 PA 和 PT 两个例子，并把日志保存到 logs/
# 用法 (在 WSL 内):
#   cd /mnt/c/Users/51418/Desktop/pamc-gpu
#   bash run_examples_gpu.sh
# 或者从 PowerShell:
#   wsl -d Ubuntu-20.04 -- bash -c "cd /mnt/c/Users/51418/Desktop/pamc-gpu && bash run_examples_gpu.sh"

set -u

# ---------- 激活 jax-gpu venv (优先 miniforge3 conda，再退到 venv) ----------
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate jax-gpu 2>/dev/null || true
elif [ -f "$HOME/jax-gpu/bin/activate" ]; then
    source "$HOME/jax-gpu/bin/activate"
fi

echo "===== Python / JAX 自检 ====="
python -c "import sys; print('python', sys.version)"
python -c "import jax; print('jax', jax.__version__); print('devices', jax.devices())"
python -c "import parmed; print('parmed', parmed.__version__)" 2>&1 || pip install -q parmed pandas pyarrow MDAnalysis
echo

mkdir -p logs results

REPO=/mnt/c/Users/51418/Desktop/pamc-gpu
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

PDB=$REPO/data/1PGB_processed.pdb
TOP=$REPO/data/1PGB.top

# ---------- 例 1: PA 中性表面 (psi0=0, 仅 vdW) ----------
echo "===== [1/2] PA  continuum surface  psi0=0 (纯 vdW) ====="
python -m ptmc.run \
    --pdb "$PDB" --top "$TOP" \
    --surface-type continuum \
    --rho-s 30.0 --c6-surf 1.0 --c12-surf 1.0 \
    --lambda-d 0.785 --z-min 0.15 --psi0 0.0 \
    --sampler pa --pa-n-walkers 256 --pa-T-start 3000.0 \
    --pa-target-ess 0.7 --pa-max-steps 200 \
    --n-steps 200 --temperature 300.0 --seed 42 \
    --n-clusters 2 --top-k 2 \
    --output results/pa_1pgb_neutral.parquet \
    2>&1 | tee logs/pa_neutral.log
PA_RC=${PIPESTATUS[0]}
echo "PA neutral exit code: $PA_RC"
echo

# ---------- 例 2: PT 带电表面 (psi0=-5 kJ/mol/e) ----------
echo "===== [2/2] PT  continuum surface  psi0=-5 (吸引正电) ====="
python -m ptmc.run \
    --pdb "$PDB" --top "$TOP" \
    --surface-type continuum \
    --rho-s 30.0 --c6-surf 1.0 --c12-surf 1.0 \
    --lambda-d 0.785 --z-min 0.15 --psi0 -5.0 \
    --sampler pt --pt-n-replicas 6 --pt-T-min 250.0 --pt-T-max 600.0 \
    --pt-n-rounds 50 --pt-n-sweep 50 \
    --n-steps 200 --temperature 300.0 --seed 42 \
    --n-clusters 2 --top-k 2 \
    --output results/pt_1pgb_charged.parquet \
    2>&1 | tee logs/pt_charged.log
PT_RC=${PIPESTATUS[0]}
echo "PT charged exit code: $PT_RC"
echo

echo "===== 汇总 ====="
echo "PA exit = $PA_RC   PT exit = $PT_RC"
ls -lh results/ 2>/dev/null
