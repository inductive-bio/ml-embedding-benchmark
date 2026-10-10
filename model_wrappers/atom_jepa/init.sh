if $INSTALL_DEP; then
    # PyTorch must already be installed. ATOM_JEPA_EXTRAS adds cuEquivariance
    # GPU kernels, e.g. ATOM_JEPA_EXTRAS=rdkit,cu12.
    pip install "atom-jepa[${ATOM_JEPA_EXTRAS:-rdkit}]==0.3.0"
fi

export HYDRA_EXPERIMENT=atom_jepa
