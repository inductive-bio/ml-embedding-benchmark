import logging as log

import numpy as np
from mordred import Calculator, descriptors as mordred_descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors

from src.common.conformers import best_mmff94s_conformer, first_etkdg_conformer
from src.common.types import SmilesEmbedder
from rdkit_3d import N_3D_FEATURES, compute_rdkit_3d

CONFORMER_FNS = {
    "etkdg": first_etkdg_conformer,
    "mmff94s": best_mmff94s_conformer,
}

# These graph-theoretic descriptors grow combinatorially with ring/path count
# and are effectively unbounded (RDKit's Ipc reached 2.86e17 on one compound
# in data/errors/reg_openadmet_expansionrx_LogD vs <=3.3e3 for every other
# rdkit2d column). StandardScaler can't tame that heavy a tail, so Ridge
# extrapolated to predictions in the tens of millions on structurally unusual
# test compounds. Signed-log1p compresses the tail while leaving
# already-reasonable descriptors elsewhere untouched.
UNBOUNDED_RDKIT2D_DESCRIPTORS = {"Ipc"}
UNBOUNDED_MORDRED_DESCRIPTORS = {"VR1_A", "VR2_A", "DetourIndex"}


def _signed_log1p(x: float) -> float:
    return float(np.sign(x) * np.log1p(np.abs(x)))


def _row_or_nan(compute_one, smi: str, dim: int) -> np.ndarray:
    try:
        row = np.asarray(list(compute_one(smi)), dtype=np.float64)
        # A successful computation can still carry individual per-descriptor
        # NaNs for legitimately inapplicable descriptors (Mordred's
        # element-conditional descriptors; degenerate 3D shape descriptors for
        # very small/ionic molecules) -- true of nearly every real molecule
        # for *some* descriptor. remove_failed_embeddings() (src/common/types.py)
        # drops any row with any NaN, so leaving these as NaN would drop most
        # real molecules -- zero them out instead, distinct from a hard
        # failure (exception below), which still NaN-fills the whole row so
        # that row is dropped as intended.
        return np.nan_to_num(row, nan=0.0)
    except Exception as e:
        log.warning(f"Failed to embed '{smi}': {type(e).__name__}: {e}")
        return np.full(dim, np.nan)


class DescriptorEmbedder(SmilesEmbedder):
    """Common forward() loop: per-molecule descriptor computation, NaN-filled
    on failure rather than dropped (matches EmbeddedDataset.remove_failed_embeddings()'s
    expectations, same as every other wrapper in this repo).
    """

    def __init__(self, name: str, dim: int):
        self._name = name
        self._dim = dim

    def _compute_one(self, smiles: str) -> np.ndarray:
        raise NotImplementedError

    def forward(self, smiles) -> np.ndarray:
        rows = [_row_or_nan(self._compute_one, smi, self._dim) for smi in smiles]
        return np.array(rows, dtype=np.float32)

    @property
    def name(self):
        return self._name

    @property
    def device_used(self):
        return "cpu"


class RDKit2DEmbedder(DescriptorEmbedder):
    """217 raw rdkit.Chem.Descriptors values -- no conformer needed."""

    def __init__(self):
        self._funcs = Descriptors.descList
        self._unbounded_idx = [i for i, (name, _) in enumerate(self._funcs) if name in UNBOUNDED_RDKIT2D_DESCRIPTORS]
        super().__init__("rdkit2d", len(self._funcs))

    def _compute_one(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Could not parse SMILES")
        values = [func(mol) for _, func in self._funcs]
        for i in self._unbounded_idx:
            values[i] = _signed_log1p(values[i])
        return values


class Mordred2DEmbedder(DescriptorEmbedder):
    """Mordred 2D-only descriptors -- no conformer needed."""

    def __init__(self):
        self._calc = Calculator(mordred_descriptors, ignore_3D=True)
        self._unbounded_idx = [i for i, d in enumerate(self._calc.descriptors) if str(d) in UNBOUNDED_MORDRED_DESCRIPTORS]
        super().__init__("mordred2d", len(self._calc))

    def _compute_one(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Could not parse SMILES")
        values = list(self._calc(mol).fill_missing())
        for i in self._unbounded_idx:
            values[i] = _signed_log1p(values[i])
        return values


class RDKit3DEmbedder(DescriptorEmbedder):
    """912 RDKit 3D shape/WHIM/GETAWAY/AUTOCORR3D/RDF/MORSE descriptors,
    computed on a single conformer at the given level of theory.
    """

    def __init__(self, conformer_level: str):
        self._conformer_fn = CONFORMER_FNS[conformer_level]
        super().__init__(f"rdkit3d_{conformer_level}", N_3D_FEATURES)

    def _compute_one(self, smiles):
        mol = self._conformer_fn(smiles)
        return compute_rdkit_3d(mol)


class Mordred3DEmbedder(DescriptorEmbedder):
    """Mordred 2D+3D descriptors, computed on a single conformer at the given
    level of theory.
    """

    def __init__(self, conformer_level: str):
        self._conformer_fn = CONFORMER_FNS[conformer_level]
        self._calc = Calculator(mordred_descriptors, ignore_3D=False)
        self._unbounded_idx = [i for i, d in enumerate(self._calc.descriptors) if str(d) in UNBOUNDED_MORDRED_DESCRIPTORS]
        super().__init__(f"mordred3d_{conformer_level}", len(self._calc))

    def _compute_one(self, smiles):
        mol = self._conformer_fn(smiles)
        values = list(self._calc(mol).fill_missing())
        for i in self._unbounded_idx:
            values[i] = _signed_log1p(values[i])
        return values


class Mordred2DRDKit3DEmbedder(DescriptorEmbedder):
    """Concatenation of Mordred 2D-only + the 912 RDKit 3D descriptors, the
    latter computed on a single conformer at the given level of theory. A
    NaN in either sub-embedding's row propagates to the concatenated row.
    """

    def __init__(self, conformer_level: str):
        self._mordred2d = Mordred2DEmbedder()
        self._rdkit3d = RDKit3DEmbedder(conformer_level)
        super().__init__(f"mordred2d_rdkit3d_{conformer_level}", self._mordred2d._dim + self._rdkit3d._dim)

    def _compute_one(self, smiles):
        # Exceptions here must propagate rather than being caught: the caller
        # (forward(), via _row_or_nan) needs to see a hard failure as an
        # exception so it NaN-fills the whole combined row. Catching it here
        # and returning a partial-NaN array instead would make it look like a
        # *successful* computation with legitimately-missing descriptors,
        # which _row_or_nan then zero-fills -- silently turning a hard
        # failure into a fake all-zero row.
        mordred_row = self._mordred2d._compute_one(smiles)
        rdkit_row = self._rdkit3d._compute_one(smiles)
        return np.concatenate([np.asarray(list(mordred_row)), np.asarray(list(rdkit_row))])


_EMBEDDERS = {
    "rdkit2d": lambda: RDKit2DEmbedder(),
    "mordred2d": lambda: Mordred2DEmbedder(),
    "rdkit3d_etkdg": lambda: RDKit3DEmbedder("etkdg"),
    "rdkit3d_mmff94s": lambda: RDKit3DEmbedder("mmff94s"),
    "mordred3d_etkdg": lambda: Mordred3DEmbedder("etkdg"),
    "mordred3d_mmff94s": lambda: Mordred3DEmbedder("mmff94s"),
    "mordred2d_rdkit3d_etkdg": lambda: Mordred2DRDKit3DEmbedder("etkdg"),
    "mordred2d_rdkit3d_mmff94s": lambda: Mordred2DRDKit3DEmbedder("mmff94s"),
}


def get_embedder(name, **_kwargs):
    key = name.lower()
    if key not in _EMBEDDERS:
        raise ValueError(f"Unknown embedder {name}. Available: {sorted(_EMBEDDERS)}")
    return _EMBEDDERS[key]()
