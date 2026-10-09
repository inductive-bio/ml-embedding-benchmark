"""RDKit-backed molecular utilities for sand."""
# Only the featurization-relevant symbol is exported eagerly so that importing
# this package (and the default SMILES->graph path) stays minimal.
from .molecule import RDMolecule

__all__ = [
    "RDMolecule",
]
