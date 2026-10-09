"""Shared pytest fixtures for the sand test-suite."""

from pathlib import Path

import pytest

# A handful of small, valid molecules covering: chain, aromatic ring,
# carbonyl/charge-prone groups and a heteroatom.
SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]


@pytest.fixture
def smiles_list():
    return list(SMILES)


@pytest.fixture
def make_small_sand_model():
    """Factory that builds a *small* but structurally faithful ``SANDModel``.

    The model is constructed from ``sand/config/compression.yml`` (identical
    ``class_path``s and the same quantizer / encoder / output classes), but with
    the heavy dimensions shrunk so it is CPU-fast and memory-light. The
    state-dict *key names* (encoder submodules, IVF/PQ residual-VQ codebook
    buffers) are independent of these size overrides, so a model built here
    still exercises the exact checkpoint-locked key surface.

    Imports are deferred into the factory so that collecting this conftest does
    not require torch/lightning to be installed.
    """
    import copy

    import yaml
    from jsonargparse import ArgumentParser

    from sand.model import SANDModel

    cfg_path = (
        Path(__file__).resolve().parents[1] / "sand" / "config" / "compression.yml"
    )
    with open(cfg_path) as fh:
        base_cfg = yaml.safe_load(fh)

    def _make():
        cfg = copy.deepcopy(base_cfg)
        backbone_args = cfg["model"]["backbone"]["init_args"]
        # Shrink the expensive dims; key names are unaffected.
        backbone_args["emb_dim"] = 64
        backbone_args["encoder_hidden_dim"] = 64
        backbone_args["encoder_num_layers"] = 2
        backbone_args["num_pq_quantizers"] = 4
        backbone_args["ivf_codebook_size"] = 64

        parser = ArgumentParser()
        parser.add_class_arguments(SANDModel, "model")
        parsed = parser.parse_object(
            {"model": cfg["model"]}, _skip_validation=True, _skip_required=True
        )
        parsed = parser.instantiate(parsed)
        return parsed.model

    return _make

