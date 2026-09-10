"""Checkpoint key-stability + state-dict round-trip guard (WP-10).

The publication claim that *checkpoints load* hinges on the checkpoint-locked
tensor/buffer names staying stable: the encoder submodules and the IVF/PQ
codebooks. This test builds the ``SANDModel`` defined by
``compression.yml`` (with shrunk dims, via the ``make_small_sand_model`` fixture)
and asserts:

* the checkpoint-locked module prefixes are present in ``state_dict()`` with
  stable keys (``backbone.encoder*`` and the ``backbone.ivfpq*`` codebooks);
* a ``state_dict()`` round-trips into a freshly-built model with identical keys
  and shapes, and ``load_state_dict(..., strict=True)`` reports no
  missing/unexpected keys and actually copies the values.

No real checkpoint file is required: a fresh ``state_dict`` is saved and
reloaded into a second freshly-built model, which guards the same key surface.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")


def test_state_dict_roundtrip_and_locked_keys(make_small_sand_model):
    model = make_small_sand_model()
    sd = model.state_dict()
    keys = set(sd.keys())

    # Checkpoint-locked module prefixes must be present and stably named.
    assert any(k.startswith("backbone.encoder") for k in keys), (
        "encoder submodule keys missing from state_dict"
    )
    assert any(k.startswith("backbone.ivfpq") for k in keys), (
        "quantizer (ivfpq) keys missing from state_dict"
    )
    expected_quantizer_buffers = {"centroids", "ema_counts", "ema_sums"}
    found_quantizer_buffers = {
        key.rsplit(".", maxsplit=1)[-1]
        for key in keys
        if key.startswith("backbone.ivfpq")
    }
    assert expected_quantizer_buffers <= found_quantizer_buffers, (
        "centroid-bank buffers missing from backbone.ivfpq"
    )

    # Round-trip into a second freshly-built model: keys + shapes must match.
    fresh = make_small_sand_model()
    fresh_keys = set(fresh.state_dict().keys())
    assert fresh_keys == keys, "state_dict key set is not stable across builds"
    for k in keys:
        assert fresh.state_dict()[k].shape == sd[k].shape, f"shape mismatch for {k}"

    incompatible = fresh.load_state_dict(sd, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys

    # Values were actually loaded.
    reloaded = fresh.state_dict()
    for k in keys:
        assert torch.equal(reloaded[k], sd[k]), f"value not loaded for {k}"
