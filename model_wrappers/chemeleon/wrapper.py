import logging as log
from os.path import join

import numpy as np
import torch
from hydra.utils import get_original_cwd

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models.model import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN

from src.common.types import SmilesEmbedder
from src.common.utils import batch, get_device

# Local checkpoint filenames, as downloaded by init.sh from the
# ensemble-aware-admet Modal volume (see that repo's
# src/models/chemeleon_3d.py / chemeleon_finetune.py for the pretraining code).
_VARIANTS = {
    "chemeleon_original": "chemeleon_mp.pt",
    "chemeleon_pubchem_3d_mean": "chemeleon_pubchem_3d_mean.ckpt",
    "chemeleon_pubchem_3d_boltzmann": "chemeleon_pubchem_3d_boltzmann.ckpt",
    "chemeleon_pubchem_both_mean": "chemeleon_pubchem_both_mean.ckpt",
    "chemeleon_pubchem_both_boltzmann": "chemeleon_pubchem_both_boltzmann.ckpt",
    "chemeleon_chembl3d_both_boltzmann": "chemeleon_chembl3d_both_boltzmann.ckpt",
    "chemeleon_geomdrugs_both_boltzmann": "chemeleon_geomdrugs_both_boltzmann.ckpt",
}


def _load_mpnn(checkpoint_path: str, map_location) -> MPNN:
    """Load either checkpoint format into a full MPNN, so `.fingerprint()` can be
    called identically regardless of which one this is.

    - Lightning-format checkpoints (this project's own pretraining runs; carry
      a "pytorch-lightning_version" key) load directly via
      `MPNN.load_from_checkpoint` -- the base class, not the `ChemeleonMPNN`
      training subclass, since the latter adds no inference-time behavior.
      All were trained with `agg=MeanAggregation()` and `batch_norm=False`.
    - The raw encoder-only checkpoint (the original externally-published
      Chemeleon weights) has no agg/predictor/bn to load, so this builds a
      fresh `MeanAggregation()` and a throwaway `RegressionFFN` (never
      invoked -- only `.fingerprint()` is ever called) purely to satisfy
      `MPNN.__init__`.
    """
    raw = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if "pytorch-lightning_version" in raw:
        return MPNN.load_from_checkpoint(checkpoint_path, map_location=map_location)

    mp = BondMessagePassing(**raw["hyper_parameters"])
    mp.load_state_dict(raw["state_dict"])
    predictor = RegressionFFN(n_tasks=1, input_dim=mp.output_dim)
    return MPNN(mp, MeanAggregation(), predictor)


class ChemeleonEmbedder(SmilesEmbedder):
    def __init__(self, name: str):
        self._model_name = name
        self._device = get_device()

        checkpoint_path = join(
            get_original_cwd(), "model_wrappers", "chemeleon", "checkpoints", _VARIANTS[name]
        )
        self._model = _load_mpnn(checkpoint_path, self._device).eval().to(self._device)
        self._dim = self._model.message_passing.output_dim

    def _encode_chunk(self, smiles_chunk):
        dataset = MoleculeDataset([MoleculeDatapoint.from_smi(smi) for smi in smiles_chunk])
        loader = build_dataloader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        bmg, V_d, X_d, *_ = next(iter(loader))
        bmg.to(self._device)
        with torch.no_grad():
            emb = self._model.fingerprint(bmg, V_d, X_d)
        return emb.cpu().numpy()

    def _safe_batch(self, smiles_batch):
        try:
            return self._encode_chunk(smiles_batch)
        except Exception as e:
            # MoleculeDatapoint.from_smi() raises on the first unparsable
            # SMILES in a chunk, taking every other molecule down with it --
            # fall back to encoding one at a time so only the bad ones get NaN'd.
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
        return np.concatenate([self._safe_batch(b) for b in batch(smiles, 256)], axis=0)

    @property
    def name(self):
        return self._model_name

    @property
    def device_used(self):
        return self._device.type.split(':')[0]


def get_embedder(name, **_kwargs) -> SmilesEmbedder:
    if name in _VARIANTS:
        return ChemeleonEmbedder(name)
    raise ValueError(f"Unknown embedder: {name}. Available: {sorted(_VARIANTS)}")
