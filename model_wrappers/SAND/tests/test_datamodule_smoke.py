"""End-to-end smoke test for the graph-pair datamodule collation.

Builds a small :class:`MolecularGraphCollection` from real RDKit-featurized
molecules, wraps it in :class:`SANDMolecularGraphDataset`, and checks that
``collate_fn`` yields a batch with the expected feature dimensionality and a
correctly shaped ``score`` tensor. Exercising the datamodule with the RDKit
backend confirms the public-release path produces model-ready batches.
"""

import numpy as np
import torch

from sand.data.datamodule import SANDMolecularGraphDataset
from sand.mol_utils.graph import MolecularGraphCollection
from sand.mol_utils.rdkit.molecule import RDMolecule


def _build_collection(smiles):
    graphs = [RDMolecule.from_smiles(smi).to_graph() for smi in smiles]

    xs, edge_indices, edge_attrs = [], [], []
    node_offsets, edge_offsets = [0], [0]
    for g in graphs:
        x = np.asarray(g.x)
        ei = np.asarray(g.edge_index)
        ea = np.asarray(g.edge_attr)
        xs.append(x)
        # edge_index stays graph-local (0-based); the collection slices per-graph.
        edge_indices.append(ei)
        edge_attrs.append(ea)
        node_offsets.append(node_offsets[-1] + x.shape[0])
        edge_offsets.append(edge_offsets[-1] + ea.shape[0])

    x = np.concatenate(xs, axis=0)
    edge_index = np.concatenate(edge_indices, axis=1)
    edge_attr = np.concatenate(edge_attrs, axis=0)

    return MolecularGraphCollection(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_offsets=np.asarray(node_offsets),
        edge_offsets=np.asarray(edge_offsets),
    )


def test_datamodule_collate(smiles_list):
    collection = _build_collection(smiles_list)

    query_idxs = np.array([0, 1])
    match_idxs = np.array([2, 3])
    scores = np.array([0.8, 0.2], dtype=np.float32)

    dataset = SANDMolecularGraphDataset(
        collection, query_idxs, match_idxs, scores
    )
    assert len(dataset) == 2

    batch = dataset.collate_fn([dataset[0], dataset[1]])

    assert batch.x.shape[1] == 16
    assert batch.edge_attr.shape[1] == 5
    assert batch["score"].shape == (2,)
    assert torch.allclose(batch["score"], torch.tensor(scores))
    assert batch["batch_size"] == 2
