#!/bin/bash

# GRIT (monroe/model/grit.py) only uses torch_geometric.utils.scatter -- no
# compiled torch-scatter/torch-sparse/torch-cluster extensions needed, unlike
# minimol/KERMT, so there's no exact (torch, cuda) wheel combo to pin against.
if $INSTALL_DEP; then
    pip install torch==2.5.1 torch_geometric==2.6.1
    pip install -e .
fi

export HYDRA_EXPERIMENT=monroe
