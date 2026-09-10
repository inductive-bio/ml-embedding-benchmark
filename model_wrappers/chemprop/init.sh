#!/bin/bash

if $INSTALL_DEP; then
    # chemprop's pyproject.toml declares torch>=2.1 with no upper bound, so pip
    # would otherwise resolve the newest wheel (needs a newer CUDA driver than
    # some GPU nodes have). Pin to the same version already verified against
    # chemprop==2.2.4 in model_wrappers/chemeleon/init.sh.
    pip install torch==2.5.1
    pip install chemprop==2.2.4
fi
