#!/bin/bash

if $INSTALL_DEP; then
    pip install unimol_tools --upgrade
    pip install huggingface_hub
fi

export HYDRA_EXPERIMENT=unimol
