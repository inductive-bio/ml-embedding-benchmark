import torch

from sand.nn.quantizer import IVFPQ, ProductQuantizer


def test_product_quantizer_uses_explicit_ema_and_preserves_gradients():
    quantizer = ProductQuantizer(
        input_dim=12,
        num_sub_quantizer=3,
        codebook_size=8,
        rotation_trick=False,
    )
    quantizer.train()
    embeddings = torch.randn(5, 12, requires_grad=True)

    quantized, indices = quantizer(embeddings)
    centroids_before_update = quantizer.codebooks.clone()
    quantized.square().mean().backward()

    assert quantized.shape == embeddings.shape
    assert indices.shape == (5, 3)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert torch.equal(quantizer.codebooks, centroids_before_update)

    quantizer.ema_update(embeddings.detach(), indices)

    assert not torch.equal(quantizer.codebooks, centroids_before_update)


def test_ivfpq_retains_the_training_and_indexing_contract():
    quantizer = IVFPQ(
        input_dim=12,
        num_ivf_quantizer=2,
        num_pq_quantizer=3,
        ivf_codebook_size=4,
    )
    embeddings = torch.randn(4, 12)

    result = quantizer(embeddings)
    reconstructed, ivf_ids, pq_indices, ivf_values, pq_values, raw_ivf_ids, residuals = result

    assert len(result) == 7
    assert reconstructed.shape == embeddings.shape
    assert ivf_ids.shape == (4,)
    assert pq_indices.shape == (4, 3)
    assert raw_ivf_ids.shape == (4, 2)
    assert torch.allclose(reconstructed, ivf_values + pq_values)
    assert torch.allclose(residuals, embeddings - ivf_values.detach())