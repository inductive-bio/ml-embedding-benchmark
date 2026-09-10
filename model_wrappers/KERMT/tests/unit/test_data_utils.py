# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kermt.util.utils data-loading helpers (filter_invalid_smiles)."""

from kermt.data import MoleculeDatapoint, MoleculeDataset
from kermt.util.utils import filter_invalid_smiles


def _dataset(smiles_list):
    """Build a MoleculeDataset from bare SMILES (no args -> no feature generation)."""
    return MoleculeDataset([MoleculeDatapoint(line=[smi]) for smi in smiles_list])


def test_filter_invalid_smiles_drops_unparseable_without_crashing():
    """Regression: SMILES that RDKit parses to None must be skipped, not crash.

    Before the fix, ``Chem.MolFromSmiles`` returning None led to
    ``None.GetNumHeavyAtoms()`` raising AttributeError. These inputs return
    None in modern RDKit: a syntax error ('C1CC1C(invalid') and a
    valence-invalid molecule ('C(C)(C)(C)(C)C', pentavalent carbon). The empty
    string is dropped by the pre-existing check.
    """
    data = _dataset(["c1ccccc1", "C1CC1C(invalid", "", "C(C)(C)(C)(C)C", "n1ccccc1"])
    filtered = filter_invalid_smiles(data)
    assert filtered.smiles() == ["c1ccccc1", "n1ccccc1"]


def test_filter_invalid_smiles_passes_valid_through_unchanged():
    """All-valid input is returned unchanged and in order."""
    data = _dataset(["CCO", "c1ccccc1", "CC(=O)O"])
    filtered = filter_invalid_smiles(data)
    assert filtered.smiles() == ["CCO", "c1ccccc1", "CC(=O)O"]


def test_filter_invalid_smiles_drops_zero_heavy_atom():
    """A SMILES that parses but has zero heavy atoms is filtered out."""
    data = _dataset(["[H][H]", "CCO"])
    filtered = filter_invalid_smiles(data)
    assert filtered.smiles() == ["CCO"]
