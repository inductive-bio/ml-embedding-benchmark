"""Tests for blam_analysis.compute_specialization's posterior-conditional draw
of u_{.e} (the specialization BLUP plus its previously-dropped conditional
variance)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import blam_analysis  # noqa: E402
from blam_analysis import compute_specialization  # noqa: E402


def _random_psd(n_e, rng, scale=1.0):
    M = rng.normal(size=(n_e, n_e))
    return scale * (M @ M.T) / n_e


class _FakeStat:
    def __init__(self, name, y_hat, V):
        self.name = name
        self.embeddings = [f"emb{i}" for i in range(len(y_hat))]
        self.y_hat = y_hat
        self.V = V


def _samples_and_stat(n_e, S, omega, V, r):
    """mu_e=0, a_e=1, y_hat_e=0 makes r_{.e} == theta_samples exactly, so
    callers can inject an arbitrary r_{.e} array `r` of shape (S, n_e)."""
    stat = _FakeStat("e", np.zeros(n_e), V)
    stat.theta_samples = r
    samples = {"mu__e": np.zeros(S), "a__e": np.ones(S), "omega__e": np.full(S, omega)}
    return samples, stat


def test_marginal_consistency():
    """Law of total variance: if r_{.e} is drawn from its correct unconditional
    marginal N(0, omega_e^2*I + V_e), the empirical variance of the resulting
    posterior draws of u recovers omega_e^2 (mean-variance + average
    conditional-variance = omega_e^2 elementwise in V_e's eigenbasis)."""
    rng = np.random.default_rng(0)
    n_e, S, omega = 5, 200_000, 0.7
    V = _random_psd(n_e, rng, scale=0.3)
    r = rng.multivariate_normal(np.zeros(n_e), omega**2 * np.eye(n_e) + V, size=S)
    samples, stat = _samples_and_stat(n_e, S, omega, V, r)

    u = compute_specialization(samples, [stat], seed=2)["e"]
    assert np.allclose(u.var(axis=0), omega**2, rtol=0.05)


def test_limit_zero_noise():
    """V_e -> 0: conditional variance -> 0 and u -> r_{.e} exactly."""
    n_e, S, omega = 4, 1000, 0.5
    V = np.eye(n_e) * 1e-12
    rng = np.random.default_rng(3)
    r = rng.normal(size=(S, n_e))
    samples, stat = _samples_and_stat(n_e, S, omega, V, r)

    u = compute_specialization(samples, [stat], seed=4)["e"]
    assert np.allclose(u, r, atol=1e-4)


def test_limit_large_noise_reverts_to_prior():
    """V_e >> omega_e^2: conditional mean -> 0 and u reverts to draws from the
    N(0, omega_e^2*I) prior, decoupled from r_{.e}."""
    n_e, S, omega = 4, 200_000, 0.3
    V = np.eye(n_e) * 1e6
    rng = np.random.default_rng(5)
    r = rng.normal(size=(S, n_e))
    samples, stat = _samples_and_stat(n_e, S, omega, V, r)

    u = compute_specialization(samples, [stat], seed=6)["e"]
    assert np.allclose(u.mean(axis=0), 0.0, atol=0.02)
    assert np.allclose(u.var(axis=0), omega**2, rtol=0.05)


def test_regression_guard_mean_matches_old_plugin(monkeypatch):
    """With the noise draw z forced to zero, the new conditional mean must
    equal the old (pre-fix) plug-in BLUP omega_e^2 * Sigma_e^-1 * r_{.e}."""

    class _ZeroRNG:
        def standard_normal(self, shape):
            return np.zeros(shape)

    n_e, S, omega = 6, 50, 0.4
    rng = np.random.default_rng(7)
    V = _random_psd(n_e, rng, scale=0.4)
    r = rng.normal(size=(S, n_e))
    samples, stat = _samples_and_stat(n_e, S, omega, V, r)

    monkeypatch.setattr(blam_analysis.np.random, "default_rng", lambda seed: _ZeroRNG())
    u = compute_specialization(samples, [stat], seed=0)["e"]

    Sigma_e = omega**2 * np.eye(n_e) + V
    g_e = np.linalg.solve(Sigma_e, r[:, :, None])[:, :, 0]
    u_old = omega**2 * g_e
    assert np.allclose(u, u_old, atol=1e-8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
