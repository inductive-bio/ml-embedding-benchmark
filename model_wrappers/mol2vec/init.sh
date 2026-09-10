#!/bin/bash

if $INSTALL_DEP; then
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
    pip install torch==2.4.0
    pip install deepchem==2.8.0 git+https://github.com/samoturk/mol2vec@850d944d5f48a58e26ed0264332b5741f72555aa
fi

export HYDRA_EXPERIMENT=mol2vec