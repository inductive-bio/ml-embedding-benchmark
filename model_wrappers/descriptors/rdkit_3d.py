"""912-dim RDKit 3D descriptor featurizer, operating on a Chem.Mol that
already carries one conformer.

Ported from ensemble-aware-admet/src/models/rdkit_3d.py (RDKit3DFeaturizer).
"""
import numpy as np
from rdkit.Chem import Descriptors3D, rdMolDescriptors

_SHAPE_FUNCS = [
    Descriptors3D.Asphericity,
    Descriptors3D.Eccentricity,
    Descriptors3D.InertialShapeFactor,
    Descriptors3D.NPR1,
    Descriptors3D.NPR2,
    Descriptors3D.PMI1,
    Descriptors3D.PMI2,
    Descriptors3D.PMI3,
    Descriptors3D.RadiusOfGyration,
    Descriptors3D.SpherocityIndex,
]

_N_SHAPE = len(_SHAPE_FUNCS)  # 10
_N_PBF = 1
_N_WHIM = 114
_N_GETAWAY = 273
_N_AUTOCORR3D = 80
_N_RDF = 210
_N_MORSE = 224
N_3D_FEATURES = _N_SHAPE + _N_PBF + _N_WHIM + _N_GETAWAY + _N_AUTOCORR3D + _N_RDF + _N_MORSE  # 912


def compute_rdkit_3d(mol) -> np.ndarray:
    shape = np.array([f(mol) for f in _SHAPE_FUNCS], dtype=np.float32)
    pbf = np.array([rdMolDescriptors.CalcPBF(mol)], dtype=np.float32)
    whim = np.array(rdMolDescriptors.CalcWHIM(mol), dtype=np.float32)
    getaway = np.array(rdMolDescriptors.CalcGETAWAY(mol), dtype=np.float32)
    autocorr3d = np.array(rdMolDescriptors.CalcAUTOCORR3D(mol), dtype=np.float32)
    rdf = np.array(rdMolDescriptors.CalcRDF(mol), dtype=np.float32)
    morse = np.array(rdMolDescriptors.CalcMORSE(mol), dtype=np.float32)
    return np.concatenate([shape, pbf, whim, getaway, autocorr3d, rdf, morse])
