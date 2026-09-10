#!/bin/bash

if $INSTALL_DEP; then
    # graphium's torch-sparse/torch-cluster/torch-scatter deps only ship
    # prebuilt wheels per exact (torch, cuda) combo -- pin torch to the
    # combo minimol's own README validates against rather than whatever
    # base_requirements.txt resolved.
    pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
    pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
    # scipy pinned <1.13: graphium's fake-graph dim-inference path builds an
    # adjacency matrix with an unrequested float16 dtype, which only scipy
    # <1.13's lenient coo_matrix dtype handling accepts -- see
    # scripts/modal_embed_minimol.py for the full explanation.
    pip install graphium==2.4.7 scipy==1.11.4 hydra-core
    # editable install so pkg_resources.resource_filename() and
    # hydra.initialize()'s relative config path (both used by
    # minimol/model.py) resolve against this checkout's ckpts/.
    pip install -e .
fi

export HYDRA_EXPERIMENT=minimol
