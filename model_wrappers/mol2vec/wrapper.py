from deepchem.feat import Mol2VecFingerprint
import logging as log
import numpy as np
from src.common.types import SmilesEmbedder


class Mol2VecFingerprintEmbedder(SmilesEmbedder):
    def __init__(self):
        self._embedder = Mol2VecFingerprint()  # loads pretrained weights from the original paper by default
        
    def forward(self, smiles):
        embeddings = self._embedder.featurize(smiles)
        
        if embeddings.ndim == 2:
            # No molecule failed -- featurize() returns a clean (n_molecules,
            # 300) array only when every row succeeded; a partial failure
            # instead returns a ragged (n_molecules,)-shaped object array
            # mixing (300,) and (0,) rows.
            return embeddings
        
        log.warning("Some molecules failed")
        
        shape = None
        for i in range(len(embeddings)):
            if embeddings[i].shape[0] > 0:
                shape = embeddings[i].shape
                break

        nan_embed = np.array([float('nan')] * shape[0])

        for j in range(len(embeddings)):
            if embeddings[j].shape[0] == 0:
                embeddings[j] = nan_embed

        return np.stack(embeddings, axis=0)
    
    @property
    def name(self):
        return "mol2vec"
    
    @property
    def device_used(self):
        return "cpu"

    
def get_embedder(name, **_kwargs) -> SmilesEmbedder:
    if name == "mol2vec":
        return Mol2VecFingerprintEmbedder()
    else:
        raise ValueError(f"Unknown embedder: {name}")
