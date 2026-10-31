#!/bin/bash

if $INSTALL_DEP; then
    # chemprop's pyproject.toml declares torch>=2.1 with no upper bound, so pip
    # would otherwise resolve the newest wheel (needs a newer CUDA driver than
    # some GPU nodes have -- same issue SAND's init.sh hit). Pin to the version
    # verified against chemprop==2.2.4 and this checkpoint.
    pip install torch==2.5.1
    pip install chemprop==2.2.4
fi

mkdir -p checkpoints

# The CheMeleon authors' published encoder weights (Burns et al. 2025).
if [ ! -f "./checkpoints/chemeleon_mp.pt" ]; then
    curl -fL -o ./checkpoints/chemeleon_mp.pt https://zenodo.org/records/15460715/files/chemeleon_mp.pt
fi

export HYDRA_EXPERIMENT=chemeleon
