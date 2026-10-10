import logging as log

import numpy as np
import torch

from atom_jepa import AtomJEPA

from src.common.types import SmilesEmbedder
from src.common.utils import batch


class AtomJEPAEmbedder(SmilesEmbedder):
    def __init__(self):
        self._model = AtomJEPA.from_pretrained("molecules").eval()
        self._dim = self._model.embedding_dim

    @torch.no_grad()
    def forward_batch(self, smiles_batch):
        return self._model.embed(smiles_batch).float().cpu().numpy()

    def safe_batch(self, smiles_batch):
        try:
            return self.forward_batch(smiles_batch)
        except Exception as e:
            log.warning(f"Batch failed, processing each molecule individually: {e}")
            results = []
            for i, smile in enumerate(smiles_batch):
                try:
                    results.append(self.forward_batch([smile]))
                except Exception as e:
                    log.error(f"Failed to embed molecule {i} of batch: {e}")
                    results.append(np.full((1, self._dim), np.nan))
            return np.concatenate(results, axis=0)

    def forward(self, smiles) -> np.ndarray:
        smiles = smiles.tolist() if not isinstance(smiles, list) else smiles
        chunks = [self.safe_batch(b) for b in batch(smiles, 128)]
        return np.concatenate(chunks, axis=0)

    @property
    def name(self):
        return "atom_jepa"

    @property
    def device_used(self):
        return "cuda" if torch.cuda.is_available() else "cpu"


def get_embedder(name, **_kwargs) -> SmilesEmbedder:
    if name == "atom_jepa":
        return AtomJEPAEmbedder()
    else:
        raise ValueError(f"Unknown embedder: {name}")
