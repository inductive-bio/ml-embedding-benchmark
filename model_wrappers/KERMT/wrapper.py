import logging as log
from os.path import join

import numpy as np
from hydra.utils import get_original_cwd

from src.common.types import SmilesEmbedder
from src.common.utils import batch, get_device

from task.extract_embeddings import extract_embeddings_batch, load_encoder_from_checkpoint

# Filename pinned by agent/config/released_model.json (nvidia/NV-KERMT-70M-v2,
# revision 7df5eb3179235fdea1e8124db73215da33d77dce). Fetched by init.sh via
# agent/scripts/fetch_released_model.py.
_CHECKPOINT_NAME = "kermt_contrastive_v2.0.pt"

# Concatenation order matches KermtFpGeneration.forward()'s "both" case
# (kermt/model/models.py) -- the paper's own canonical fingerprint, reachable
# via `main.py fingerprint --fingerprint_source both`. Reimplemented here by
# concatenating extract_embeddings_batch()'s four readouts directly (same
# encoder forward pass, same mean readout) rather than shelling out to
# main.py, since that path requires writing SMILES to an on-disk CSV per
# batch and has no in-process API.
_EMBEDDING_TYPES = ["atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond"]


class KERMTEmbedder(SmilesEmbedder):
    def __init__(self):
        self._device = get_device()
        self._device_str = self._device.type.split(":")[0]

        checkpoint_path = join(
            get_original_cwd(), "model_wrappers", "KERMT", "checkpoints", _CHECKPOINT_NAME
        )
        self._encoder, self._readout, self._args = load_encoder_from_checkpoint(
            checkpoint_path, device=self._device_str
        )
        self._dim = self._args.hidden_size * len(_EMBEDDING_TYPES)

    def _encode_chunk(self, smiles_chunk):
        # Invalid/unparsable SMILES are zero-filled by extract_embeddings_batch
        # (it canonicalizes internally and skips them in the encoder forward
        # pass); convert those to NaN here so remove_failed_embeddings() drops
        # them like every other wrapper's failure convention.
        embeddings, validity = extract_embeddings_batch(
            self._encoder, self._readout, smiles_chunk, self._args, device=self._device_str,
        )
        fp = np.concatenate([embeddings[t] for t in _EMBEDDING_TYPES], axis=1)
        fp[~np.array(validity, dtype=bool)] = np.nan
        return fp

    def _safe_batch(self, smiles_batch):
        try:
            return self._encode_chunk(smiles_batch)
        except Exception as e:
            log.warning(f"Batch failed, processing each SMILES individually: {e}")
            rows = []
            for smi in smiles_batch:
                try:
                    rows.append(self._encode_chunk([smi])[0])
                except Exception as inner_e:
                    log.error(f"Failed to embed '{smi}': {inner_e}")
                    rows.append(np.full(self._dim, np.nan))
            return np.stack(rows, axis=0)

    def forward(self, smiles):
        return np.concatenate([self._safe_batch(b) for b in batch(smiles, 64)], axis=0)

    @property
    def name(self):
        return "kermt"

    @property
    def device_used(self):
        return self._device_str


def get_embedder(name, **_kwargs) -> SmilesEmbedder:
    if name == "kermt":
        return KERMTEmbedder()
    raise ValueError(f"Unknown embedder: {name}. Available: ['kermt']")
