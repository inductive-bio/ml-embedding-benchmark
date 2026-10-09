"""RDKit featurizer sanity tests.

Asserts the RDKit graph featurizer produces the schema the model architecture
expects: 16 atom-feature columns (vocab < 62) and 5 bond-feature columns
(vocab < 16), with a consistent ``[2, E]`` edge index. This is the exact schema
expected by the model architecture, so checkpoints remain dimensionally
compatible.
"""

import pytest

from sand.mol_utils.rdkit.featurizer import (
    ATOM_FEATURE_VOCAB,
    BOND_FEATURE_VOCAB,
    NUM_ATOM_FEATURES,
    NUM_BOND_FEATURES,
)
from sand.mol_utils.graph import MolecularGraph
from sand.mol_utils.rdkit.molecule import RDMolecule


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "CC(=O)O", "CCN"])
def test_featurizer_schema(smiles):
    graph = RDMolecule.from_smiles(smiles).to_graph()

    x = graph.x
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr

    assert x.shape[1] == NUM_ATOM_FEATURES == 16
    assert edge_attr.shape[1] == NUM_BOND_FEATURES == 5
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_attr.shape[0]

    # Pre-offset column values must index into the shared embedding vocab.
    assert int(x.max()) < ATOM_FEATURE_VOCAB == 62
    if edge_attr.shape[0] > 0:
        assert int(edge_attr.max()) < BOND_FEATURE_VOCAB == 16


def test_single_atom_molecule_has_no_edges():
    graph = RDMolecule.from_smiles("C").to_graph()
    assert graph.x.shape[1] == NUM_ATOM_FEATURES
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, NUM_BOND_FEATURES)


def test_batching_preserves_schema(smiles_list):
    graphs = [RDMolecule.from_smiles(smi).to_graph() for smi in smiles_list]
    batch = MolecularGraph.from_graph_list(graphs)

    n_atoms = sum(g.x.shape[0] for g in graphs)
    n_edges = sum(g.edge_attr.shape[0] for g in graphs)

    assert batch.x.shape == (n_atoms, NUM_ATOM_FEATURES)
    assert batch.edge_attr.shape == (n_edges, NUM_BOND_FEATURES)
    assert batch.edge_index.shape == (2, n_edges)
