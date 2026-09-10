import logging as log

import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra

from minimol import Minimol

from src.common.types import SmilesEmbedder
from src.common.utils import batch


class MinimolEmbedder(SmilesEmbedder):
    def __init__(self):
        # embed.py's own @hydra.main call has already composed and resolved
        # its config by the time get_embedder() runs, but leaves Hydra's
        # single global singleton initialized -- Minimol.load_config()'s own
        # hydra.initialize() call for its packaged config conflicts with
        # that ("GlobalHydra is already initialized") unless it's cleared
        # first.
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        self._model = Minimol()
        # gnn:15 fingerprint layer is fixed-size for the released checkpoint;
        # determined via a throwaway call so safe_batch's NaN-fallback path
        # has a shape even if the very first real batch fails.
        self._dim = self._model(["C"])[0].shape[0]

    def forward_batch(self, smiles_batch):
        reprs = self._model(smiles_batch)
        return torch.stack(reprs).cpu().numpy()

    def forward_step(self, smile):
        return self._model([smile])[0].cpu().numpy()

    def safe_batch(self, smiles_batch):
        try:
            return self.forward_batch(smiles_batch)
        except Exception as e:
            log.warning(f"Batch failed, processing each smile individually: {e}")
            results = []
            for smile in smiles_batch:
                try:
                    results.append(self.forward_step(smile))
                except Exception as e:
                    log.error(f"Failed to process smile '{smile}': {e}")
                    results.append(np.full(self._dim, np.nan))
            return np.stack(results, axis=0)

    def forward(self, smiles) -> np.ndarray:
        smiles = smiles.tolist() if not isinstance(smiles, list) else smiles
        chunks = [self.safe_batch(b) for b in batch(smiles, 128)]
        return np.concatenate(chunks, axis=0)

    @property
    def name(self):
        return "minimol"

    @property
    def device_used(self):
        return "cuda" if torch.cuda.is_available() else "cpu"


def get_embedder(name, **_kwargs) -> SmilesEmbedder:
    if name == "minimol":
        return MinimolEmbedder()
    else:
        raise ValueError(f"Unknown embedder: {name}")
