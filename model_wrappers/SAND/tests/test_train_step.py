"""Single forward+loss step guard on synthetic data (WP-10).

The publication claim that *one training step runs* is guarded here: build the
``SANDModel`` from ``compression.yml`` (shrunk dims via ``make_small_sand_model``),
hand-build one tiny synthetic batch in exactly the shape
``SANDMolecularGraphDataset.collate_fn`` produces (a batched ``MolecularGraph``
of a few RDKit-featurized molecules plus a ``score`` tensor and ``batch_size``),
run the backbone forward and every output's loss, and assert the total loss is
finite and differentiable.

The loss is computed by calling each ``Output`` directly with a no-op log
function, which exercises the same code path as ``training_step`` without
needing an attached Lightning ``Trainer``. CPU-only and fast.
"""

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("lightning")
pytest.importorskip("rdkit")

from sand.data.datamodule import SANDMolecularGraphDataset  # noqa: E402
from sand.mol_utils.graph import MolecularGraphCollection  # noqa: E402
from sand.mol_utils.rdkit.molecule import RDMolecule  # noqa: E402


def _build_collection(smiles):
    graphs = [RDMolecule.from_smiles(smi).to_graph() for smi in smiles]

    xs, edge_indices, edge_attrs = [], [], []
    node_offsets, edge_offsets = [0], [0]
    for g in graphs:
        x = np.asarray(g.x)
        ei = np.asarray(g.edge_index)
        ea = np.asarray(g.edge_attr)
        xs.append(x)
        edge_indices.append(ei)
        edge_attrs.append(ea)
        node_offsets.append(node_offsets[-1] + x.shape[0])
        edge_offsets.append(edge_offsets[-1] + ea.shape[0])

    return MolecularGraphCollection(
        x=np.concatenate(xs, axis=0),
        edge_index=np.concatenate(edge_indices, axis=1),
        edge_attr=np.concatenate(edge_attrs, axis=0),
        node_offsets=np.asarray(node_offsets),
        edge_offsets=np.asarray(edge_offsets),
    )


def test_single_train_step_finite(make_small_sand_model, smiles_list):
    collection = _build_collection(smiles_list)

    n = len(smiles_list)
    # Form n query/match pairs (>= 2 pairs so the uniformity / soft-spearman
    # losses are well-defined).
    query_idxs = np.arange(n)
    match_idxs = (np.arange(n) + 1) % n
    scores = np.linspace(0.1, 0.9, n).astype(np.float32)

    dataset = SANDMolecularGraphDataset(collection, query_idxs, match_idxs, scores)
    batch = dataset.collate_fn([dataset[i] for i in range(n)])
    assert batch["batch_size"] == n

    model = make_small_sand_model()
    model.train()

    # Backbone forward: encode + quantize, populating emb / emb_quantized /
    # ivf_x / pq_x_res / x_res that the outputs consume.
    batch = model(batch)

    def _noop_log(*args, **kwargs):
        return None

    total = torch.zeros(())
    for output in model.outputs:
        total = total + output(
            batch, stage="train", log_func=_noop_log, dataloader_idx=0
        )

    assert total.requires_grad, "total loss should be differentiable in train mode"
    assert torch.isfinite(total), "training-step loss is not finite"
