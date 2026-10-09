"""Import sanity tests.

These tests assert that the core RDKit-based import surface works.
"""

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "sand",
        "sand.mol_utils",
        "sand.mol_utils.rdkit.molecule",
        "sand.mol_utils.rdkit.featurizer",
        "sand.mol_utils.pytorch",
        "sand.data.datamodule",
        "sand.index.data",
        "sand.index.index",
        "sand.model",
    ],
)
def test_core_modules_import(module):
    import importlib

    importlib.import_module(module)


def test_molecule_cls_is_rdmolecule():
    from sand.mol_utils import RDMolecule

    assert RDMolecule.from_smiles("CCO") is not None
