#!/bin/bash

if $INSTALL_DEP; then
    pip install -e ./huggingmolecules_repo/src
    pip install git+https://github.com/bp-kelley/descriptastorus@9a190343bcd3cfd35142d378d952613bcac40797
fi

export PYTHONPATH=
export HYDRA_EXPERIMENT=huggingmolecules
