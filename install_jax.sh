#!/bin/bash
set -e

export PATH="$HOME/miniforge3/bin:$PATH"
source "$HOME/miniforge3/etc/profile.d/conda.sh"

TUNA_PIP="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"

# ── 1. Activate env (already created) ────────────────────────
echo "=== [1/3] Activating jax-gpu env ==="
conda activate jax-gpu
echo "  Python: $(python --version)"

# ── 2. Install JAX with CUDA 12 (Tsinghua mirror) ────────────
echo "=== [2/3] Installing JAX CUDA12 (Tsinghua mirror) ==="
pip install --upgrade pip $TUNA_PIP
pip install "jax[cuda12]" $TUNA_PIP
pip install numpy scipy matplotlib pandas jupyter ipykernel $TUNA_PIP
echo "  Done."

# ── 3. Verify GPU ─────────────────────────────────────────────
echo "=== [3/3] Verifying JAX GPU ==="
python3 - <<'PYEOF'
import jax
print("JAX version:", jax.__version__)
print("All devices:", jax.devices())
try:
    gpu = jax.devices("gpu")
    print("GPU devices:", gpu)
    import jax.numpy as jnp
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    print("Matmul shape:", y.shape)
    print("SUCCESS: JAX GPU working!")
except Exception as e:
    print("WARNING:", e)
PYEOF

echo ""
echo "=== DONE ==="
echo "Usage:"
echo "  source ~/miniforge3/etc/profile.d/conda.sh && conda activate jax-gpu"
