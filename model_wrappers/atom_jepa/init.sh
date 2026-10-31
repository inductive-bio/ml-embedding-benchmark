if $INSTALL_DEP; then
    # atom-jepa needs PyTorch installed first. Pinned to the version the
    # published Atom-JEPA results were produced with (a CUDA 13 build, which
    # ran with the cu12 cuEquivariance kernels on Modal). ATOM_JEPA_EXTRAS adds
    # those kernels, e.g. ATOM_JEPA_EXTRAS=rdkit,cu12.
    pip install torch==2.14.1
    pip install "atom-jepa[${ATOM_JEPA_EXTRAS:-rdkit}]==0.3.0"
fi

export HYDRA_EXPERIMENT=atom_jepa
