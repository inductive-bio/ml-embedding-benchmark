import logging as log
import numpy as np

from src.common.types import SmilesEmbedder
from src.common.utils import batch
from hydra.utils import get_original_cwd
from os.path import join
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer


class MolBertEmbedder(SmilesEmbedder):
    def __init__(self, model_path: str):
        self._model_name = model_path
        self._embedder = MolBertFeaturizer(join(get_original_cwd(), "model_wrappers/molbert/weights/molbert_100epochs/checkpoints/last.ckpt"), device='cpu')
        log.info(f"Device {self._embedder.device}")

    def forward(self, smiles):
        # MolBertFeaturizer.transform() runs its whole input through BERT in a
        # single forward pass with no internal batching -- OOM-killed this
        # process on a 30GB machine with a few thousand SMILES. Chunk it like
        # every other wrapper does.
        return np.concatenate([self._embedder.transform(b)[0] for b in batch(smiles, 128)], axis=0)
    
    @property
    def name(self):
        return self._model_name
    
    
def get_embedder(name, **_kwagrs) -> MolBertEmbedder:
    return MolBertEmbedder(name)
