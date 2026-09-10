#!/bin/bash

if $INSTALL_DEP; then
  pip install rdkit mordredcommunity pandas numpy
fi

export HYDRA_EXPERIMENT=descriptors
