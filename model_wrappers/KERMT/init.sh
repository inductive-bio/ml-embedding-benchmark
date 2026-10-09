#!/bin/bash

# KERMT's own environment.yml pins cuik_molmaker>=0.3, which NVIDIA only
# publishes as a prebuilt conda package (nvidia/conda-forge channels) -- it is
# NOT on PyPI at that version (only an older 0.2 source dist exists there,
# which fails to build outside a matching CUDA/conda toolchain). That makes
# this wrapper incompatible with the plain pyenv+venv+pip flow every other
# wrapper here uses via embed_wrapper.sh. Use scripts/modal_embed_kermt.py
# instead, which builds the Modal image with `micromamba_install` directly
# from environment.yml (see that script for the real, working install path).
#
# This init.sh is kept only so embed_wrapper.sh doesn't hard-fail immediately,
# and to share the checkpoint-download step with local ad-hoc use if someone
# separately provisions a conda/micromamba env with cuik_molmaker installed
# and points PYTHONPATH at this directory themselves.
if $INSTALL_DEP; then
    pip install torch==2.9.1
    pip install rdkit==2025.9.1
    pip install scikit-learn==1.9.0 descriptastorus==2.7.0 huggingface_hub==0.36.2
    pip install hydra-core hydra-joblib-launcher joblib tqdm pandas
    pip install cuik-molmaker==0.2 || echo "WARNING: cuik-molmaker (conda-only >=0.3) not installed -- this wrapper will not run outside the Modal image built by scripts/modal_embed_kermt.py"
fi

python agent/scripts/fetch_released_model.py --out checkpoints

export HYDRA_EXPERIMENT=kermt
