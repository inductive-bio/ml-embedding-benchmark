# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MIT License

# Copyright (c) 2021 Tencent AI Lab.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import pytest

from scripts.build_vocab import build
from kermt.data.torchvocab import MolVocab
from argparse import Namespace
import os

def test_build_vocab(data_dir):
    data_path = data_dir / "smis_only.csv"
    save_dir = os.getcwd()
    args = Namespace(
        data_path=str(data_path),
        vocab_save_folder=str(save_dir),
        dataset_name="test_build_vocab",
        vocab_max_size=None,
        vocab_min_freq=1,
        vocab_format='json',
        # SMILES-vocab + mode-selection options.
        # Build atom/bond only to keep this test focused on the original assertions;
        # SMILES-vocab building has its own fixtures elsewhere.
        build_all=False,
        build_atom_bond_only=True,
        build_smiles_only=False,
        smiles_vocab_max_size=None,
        smiles_vocab_min_freq=1,
        smiles_regex_path=None,
        num_workers=1,
    )
    build(args)

    atom_vocab = MolVocab.load_vocab(os.path.join(save_dir, "test_build_vocab_atom_vocab.json"))
    bond_vocab = MolVocab.load_vocab(os.path.join(save_dir, "test_build_vocab_bond_vocab.json"))

    atom_vocab_ref = MolVocab.load_vocab(str(data_dir / "test_build_vocab_atom_vocab.json"))
    bond_vocab_ref = MolVocab.load_vocab(str(data_dir / "test_build_vocab_bond_vocab.json"))

    assert len(atom_vocab) == len(atom_vocab_ref), f"Atom vocab size mismatch: {len(atom_vocab)} != {len(atom_vocab_ref)}"
    assert len(bond_vocab) == len(bond_vocab_ref), f"Bond vocab size mismatch: {len(bond_vocab)} != {len(bond_vocab_ref)}"

    assert atom_vocab.stoi.keys() == atom_vocab_ref.stoi.keys(), f"Atom vocab keys mismatch: {atom_vocab.stoi.keys()} != {atom_vocab_ref.stoi.keys()}"
    assert bond_vocab.stoi.keys() == bond_vocab_ref.stoi.keys(), f"Bond vocab keys mismatch: {bond_vocab.stoi.keys()} != {bond_vocab_ref.stoi.keys()}"

    # Clean up generated vocab files after test
    for f in ("test_build_vocab_atom_vocab.json", "test_build_vocab_bond_vocab.json"):
        path = os.path.join(save_dir, f)
        if os.path.exists(path):
            os.remove(path)