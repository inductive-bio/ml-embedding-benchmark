#!/bin/bash

if $INSTALL_DEP; then
    # sand's pyproject.toml leaves "torch" unversioned, so pip resolves the
    # latest wheel (needs a newer CUDA driver than many GPU nodes have).
    # torch.cuda.is_available() doesn't catch this -- it only fails once a
    # tensor actually moves to the GPU. Pin to a driver-compatible release
    # (same version other wrappers here use, e.g. mol2vec).
    pip install torch==2.4.0
    # sand/__init__.py unconditionally imports SANDIndex, which needs faiss,
    # even though this wrapper only uses SANDModel.encode_smiles(). faiss-cpu
    # (not the gpu extra) satisfies that import; we never call into the
    # index/search code, so its GPU support is irrelevant here.
    pip install -e ".[cpu]"
fi

mkdir -p checkpoints
if [ ! -f "./checkpoints/sand_no_comp.ckpt" ]; then
    curl -L -o ./checkpoints/sand_no_comp.ckpt \
        https://github.com/pfizer-opensource/SAND/releases/download/v1.0.0/sand_no_comp.ckpt
fi
if [ ! -f "./checkpoints/sand_k256_m32.ckpt" ]; then
    curl -L -o ./checkpoints/sand_k256_m32.ckpt \
        https://github.com/pfizer-opensource/SAND/releases/download/v1.0.0/sand_k256_m32.ckpt
fi

export HYDRA_EXPERIMENT=sand
