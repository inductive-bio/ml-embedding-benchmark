import logging as log
from os.path import join

import numpy as np
from hydra.utils import get_original_cwd

from src.common.types import SmilesEmbedder
from src.common.utils import batch, get_device

from monroe.model.ckpt import load_ckpt
from monroe.eval.embed import embed_smiles

_CHECKPOINT_DIR = "checkpoint"


class MonroeEmbedder(SmilesEmbedder):
    def __init__(self):
        self._device = get_device()
        checkpoint_path = join(get_original_cwd(), "model_wrappers", "Monroe", _CHECKPOINT_DIR)
        # load_ckpt() calls torch.load() with no map_location override, so it
        # restores tensors onto the device they were saved from (CUDA) with
        # no CPU fallback -- matches this wrapper's intended Modal GPU path,
        # but raises on a CPU-only machine.
        self._encoder = load_ckpt(checkpoint_path).to(self._device).eval()
        # "C" is always featurizable, so this canary call is safe to use for
        # sizing the NaN-fill fallback in _safe_chunk below.
        self._dim = embed_smiles(["C"], self._encoder, device=self._device, batch_size=1)["C"].shape[0]

    def _safe_chunk(self, smiles_chunk):
        try:
            embeddings = embed_smiles(smiles_chunk, self._encoder, device=self._device, batch_size=64)
        except Exception as e:
            log.warning(f"Chunk failed, embedding each SMILES individually: {e}")
            embeddings = {}
            for smi in smiles_chunk:
                try:
                    embeddings.update(embed_smiles([smi], self._encoder, device=self._device, batch_size=1))
                except Exception as inner_e:
                    log.error(f"Failed to embed '{smi}': {inner_e}")
        # embed_smiles() silently omits SMILES it can't featurize from its
        # returned dict rather than NaN-filling them -- reinstate NaN rows
        # here so remove_failed_embeddings() drops them like other wrappers.
        return np.stack([
            embeddings[smi] if smi in embeddings else np.full(self._dim, np.nan, dtype=np.float32)
            for smi in smiles_chunk
        ], axis=0)

    def forward(self, smiles) -> np.ndarray:
        smiles = smiles.tolist() if not isinstance(smiles, list) else smiles
        chunks = [self._safe_chunk(c) for c in batch(smiles, 128)]
        return np.concatenate(chunks, axis=0)

    @property
    def name(self):
        return "monroe"

    @property
    def device_used(self):
        return self._device.type


def get_embedder(name, **_kwargs) -> SmilesEmbedder:
    if name == "monroe":
        return MonroeEmbedder()
    raise ValueError(f"Unknown embedder: {name}. Available: ['monroe']")
