"""ETKDG + MMFF conformer generation, shared by any wrapper that needs a 3D
conformer for a molecule given only its SMILES.

Exposes two levels of theory: `first_etkdg_conformer` (raw ETKDG
distance-geometry embedding, no force-field refinement) and
`best_mmff94s_conformer` (ETKDG ensemble + MMFF94s optimization, UFF fallback,
lowest-energy pick).
"""
import copy
import os

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

# The 3 descriptor families needing a conformer (rdkit3d, mordred3d,
# mordred2d_rdkit3d), each ablated across 2 levels of theory, run as separate
# embed.py invocations, so an in-memory cache wouldn't be reused between them;
# this disk cache is, keyed by each cached function's own arguments.
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "conformer_cache",
)
_memory = joblib.Memory(location=_CACHE_DIR, verbose=0)


def create_etkdg_ensemble(smiles: str, num_confs: int = 5, rms_threshold: float = 0.25) -> list:
    """Embed `num_confs` ETKDG conformers for `smiles`, pruning near-duplicates.

    Returns a list of single-conformer Chem.Mol objects (Hs added). Empty if
    embedding failed outright (e.g. degenerate/disconnected input).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Could not parse SMILES string")
    mol = Chem.AddHs(mol)

    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=42, pruneRmsThresh=rms_threshold)

    ensemble = []
    for conf in mol.GetConformers():
        new_mol = copy.deepcopy(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(Chem.Conformer(conf))
        ensemble.append(new_mol)
    return ensemble


def mmff_optimize(ensemble: list) -> tuple:
    """Optimize each conformer with MMFF94s, falling back to UFF for
    molecules MMFF94s can't parametrize. Conformers that don't converge
    within maxIts are dropped.

    Returns (converged_conformers, their kcal/mol energies). Raises
    ValueError if neither force field can be parametrized at all.
    """
    converged_ensemble = []
    energies = []
    for conf in ensemble:
        mmff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(conf, mmffVariant="MMFF94s")
        if mmff_props is None:
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(conf)
            if ff is None:
                raise ValueError("Neither MMFF94s nor UFF could be parametrized for this molecule")
        else:
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(conf, mmff_props)
            if ff is None:
                raise ValueError("MMFF94s force field could not be constructed for this molecule")

        ff.Initialize()
        if ff.Minimize(maxIts=1000) != 0:
            continue
        converged_ensemble.append(conf)
        energies.append(ff.CalcEnergy())

    return converged_ensemble, energies


def _lowest_energy_conformer(smiles: str, num_confs: int = 5, rms_threshold: float = 0.25) -> Chem.Mol:
    ensemble = create_etkdg_ensemble(smiles, num_confs=num_confs, rms_threshold=rms_threshold)
    if not ensemble:
        raise ValueError("ETKDG embedding produced no conformers")
    converged, energies = mmff_optimize(ensemble)
    if not converged:
        raise ValueError("No conformer converged during MMFF/UFF optimization")
    return converged[int(np.argmin(energies))]


best_mmff94s_conformer = _memory.cache(_lowest_energy_conformer)


def _first_etkdg_conformer(smiles: str, seed: int = 42) -> Chem.Mol:
    """Single ETKDG embedding, no force-field refinement -- the "raw ETKDG"
    level of theory. Uses one EmbedMolecule call rather than
    create_etkdg_ensemble's num_confs=5 (whose extra conformers only exist to
    give MMFF something to rank) since nothing here ranks or uses them.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Could not parse SMILES string")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise ValueError("ETKDG embedding failed to produce a conformer")
    return mol


first_etkdg_conformer = _memory.cache(_first_etkdg_conformer)
