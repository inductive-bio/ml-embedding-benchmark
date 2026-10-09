#!/bin/bash

if $INSTALL_DEP; then
    # chemprop's pyproject.toml declares torch>=2.1 with no upper bound, so pip
    # would otherwise resolve the newest wheel (needs a newer CUDA driver than
    # some GPU nodes have -- same issue SAND's init.sh hit). Pin to the version
    # already verified against chemprop==2.2.4 and these checkpoints in
    # ensemble-aware-admet's own venv.
    pip install torch==2.5.1
    pip install chemprop==2.2.4
    # Needed below to pull checkpoints from the ensemble-aware-admet Modal
    # volume -- requires a pre-authenticated "inductive-bio" modal profile on
    # whatever host runs this script; init.sh cannot perform that
    # authentication itself.
    pip install modal
fi

mkdir -p checkpoints

download() {
    local remote=$1
    local local_name=$2
    if [ ! -f "./checkpoints/$local_name" ]; then
        modal volume get ensemble-aware-admet "$remote" "./checkpoints/$local_name"
    fi
}

# All 7 checkpoints live on the `ensemble-aware-admet` Modal volume (see that
# repo's src/models/chemeleon_3d.py / chemeleon_finetune.py for the
# pretraining code). Renamed locally since the 6 Lightning checkpoints are all
# literally named "best.ckpt" on the remote volume.
download checkpoints/chemeleon/chemeleon_mp.pt                       chemeleon_mp.pt
download checkpoints/test-pubchem-mean-3d-only/best.ckpt              chemeleon_pubchem_3d_mean.ckpt
download checkpoints/test-pubchem-boltzmann-3d-only/best.ckpt         chemeleon_pubchem_3d_boltzmann.ckpt
download checkpoints/test-pubchem-3d-and-mordred-mean/best.ckpt       chemeleon_pubchem_both_mean.ckpt
download checkpoints/test-pubchem-3d-and-mordred-boltzmann/best.ckpt  chemeleon_pubchem_both_boltzmann.ckpt
download checkpoints/test-chembl3d-boltzmann-both/best.ckpt           chemeleon_chembl3d_both_boltzmann.ckpt
download checkpoints/test-geom-drugs-boltzmann-both/best.ckpt         chemeleon_geomdrugs_both_boltzmann.ckpt

export HYDRA_EXPERIMENT=chemeleon
