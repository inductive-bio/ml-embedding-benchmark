import logging as log
from os.path import join

import numpy as np
import torch
from hydra.utils import get_original_cwd

from src.common.types import SmilesEmbedder
from src.common.utils import batch, get_device

from sand.model import SANDModel

# The two checkpoints published in the SAND v1.0.0 GitHub release: a dense
# (non-quantized) encoder and one trained with IVF-PQ quantization (k=256,
# m=32). Exposed as separate model_name variants since they're genuinely
# different embeddings (dense vs. the quantized reconstruction actually used
# for FAISS search), both of dim 512.
_VARIANTS = {
    "sand": dict(checkpoint="sand_no_comp.ckpt", config="no_compression.yml", emb_key="emb"),
    "sand_quantized": dict(checkpoint="sand_k256_m32.ckpt", config="compression.yml", emb_key="emb_quantized"),
}


class SANDEmbedder(SmilesEmbedder):
    def __init__(self, name: str):
        variant = _VARIANTS[name]
        self._model_name = name
        self._emb_key = variant["emb_key"]
        self._device = get_device()

        wrapper_dir = join(get_original_cwd(), "model_wrappers", "SAND")
        checkpoint_path = join(wrapper_dir, "checkpoints", variant["checkpoint"])
        config_path = join(wrapper_dir, "sand", "config", variant["config"])

        self._model = SANDModel.load_from_checkpoint(
            checkpoint_path, config_path, map_location=self._device
        ).eval().to(self._device)
        self._dim = self._model.backbone.emb_dim

    def _safe_batch(self, smiles_batch):
        try:
            with torch.no_grad():
                emb = self._model.encode_smiles(smiles_batch, emb_key=self._emb_key, batch_size=len(smiles_batch))
            return emb.cpu().numpy()
        except Exception as e:
            # smiles_to_batch() raises on the first unparsable SMILES in the
            # batch, so one bad molecule would otherwise take out every
            # other embedding in the batch with it.
            log.warning(f"Batch failed, processing each SMILES individually: {e}")
            rows = []
            for smi in smiles_batch:
                try:
                    with torch.no_grad():
                        emb = self._model.encode_smiles([smi], emb_key=self._emb_key, batch_size=1)
                    rows.append(emb.cpu().numpy()[0])
                except Exception as inner_e:
                    log.error(f"Failed to embed '{smi}': {inner_e}")
                    rows.append(np.full(self._dim, np.nan))
            return np.stack(rows, axis=0)

    def forward(self, smiles):
        return np.concatenate([self._safe_batch(b) for b in batch(smiles, 256)], axis=0)

    @property
    def name(self):
        return self._model_name

    @property
    def device_used(self):
        return self._device.type.split(':')[0]


def get_embedder(name, **_kwargs) -> SmilesEmbedder:
    if name in _VARIANTS:
        return SANDEmbedder(name)
    raise ValueError(f"Unknown embedder: {name}. Available: {sorted(_VARIANTS)}")
