"""Monte Carlo check of the variance inflation derived in docs/bayesian_model.md
("Variance Inflation" sections): the Bradley-Terry pseudo-likelihood understates
Var(beta_hat) by a factor of (M+1)/3, M the number of models being compared.

Data-generating process: each dataset produces one Plackett-Luce ranking of all M models
(Gumbel-max representation), from which all M(M-1)/2 pairwise outcomes are read off. PL has
BT pairwise marginals, P(i > j) = sigmoid(beta_i - beta_j), so beta_hat stays consistent and
any gap between empirical and model-based variance is inflation, not bias.

Scenarios A-E fit by Fisher scoring (`fit`) and compare the analytic naive/sandwich
variances. Scenario F instead fits the same pairwise likelihood with NUTS (numpyro),
per blam_analysis.py's Bayesian BT model, to check whether the posterior itself
inherits the naive estimator's undercoverage -- at far fewer reps than A-E, since each
replicate is a full NUTS fit; see scripts/modal_pl_variance_inflation.py to run scenario
F's Monte Carlo at the same scale as A-E, fanned out across containers.

Usage:
    python scripts/pl_variance_inflation.py
"""

import os

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.linalg import helmert
from scipy.stats import norm

NOMINAL_Z = norm.ppf(0.975)
OUTPUT_PNG = "data/pl_variance_inflation/coverage.png"

# Categorical slots 1/2 from the dataviz skill's validated default palette
# (references/palette.md), same hexes scripts/visualize_blam.py already uses.
COLOR_NAIVE = "#2a78d6"  # blue
COLOR_SANDWICH = "#eb6834"  # orange
COLOR_TARGET = "#52514e"  # secondary ink, for the 95% reference line
COLOR_GRID = "#e1e0d9"
COLOR_TEXT_MUTED = "#898781"
COLOR_TEXT_PRIMARY = "#0b0b0b"
SURFACE = "#fcfcfb"


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def lap_pinv(H):
    """Pseudo-inverse of a Laplacian-type matrix whose null space is span(1)."""
    P0 = np.full(H.shape, 1.0 / H.shape[0])
    return np.linalg.inv(H + P0) - P0


def sample(beta, D, rng, coupled=True):
    """Return win counts W (M,M) and per-dataset win totals R (D,M)."""
    M = beta.size
    if coupled:
        scores = beta + rng.gumbel(size=(D, M))
        X = (scores[:, :, None] > scores[:, None, :]).astype(np.float64)
    else:
        # Control: same BT marginals, but each pair drawn independently (no transitivity).
        pi = sigmoid(beta[:, None] - beta[None, :])
        iu = np.triu_indices(M, 1)
        draws = (rng.random((D, iu[0].size)) < pi[iu]).astype(np.float64)
        X = np.zeros((D, M, M))
        X[:, iu[0], iu[1]] = draws
        X[:, iu[1], iu[0]] = 1.0 - draws
    return X.sum(axis=0), X.sum(axis=2)


def hess(beta, D):
    M = beta.size
    N = D * (1.0 - np.eye(M))
    pi = sigmoid(beta[:, None] - beta[None, :])
    v = N * pi * (1.0 - pi)
    np.fill_diagonal(v, 0.0)
    return np.diag(v.sum(axis=1)) - v


def fit(W, D, iters=100, tol=1e-11):
    """BT MLE by Fisher scoring, identified by sum(beta) = 0."""
    M = W.shape[0]
    N = D * (1.0 - np.eye(M))
    wins = W.sum(axis=1)
    beta = np.zeros(M)
    for _ in range(iters):
        pi = sigmoid(beta[:, None] - beta[None, :])
        U = wins - (N * pi).sum(axis=1)
        step = lap_pinv(hess(beta, D)) @ U
        beta = beta + step
        beta -= beta.mean()
        if np.max(np.abs(step)) < tol:
            break
    return beta


def sandwich(beta, R, D):
    """H^+ J H^+ with J estimated from per-dataset scores (clustered by dataset)."""
    pi = sigmoid(beta[:, None] - beta[None, :])
    np.fill_diagonal(pi, 0.0)
    Ud = R - pi.sum(axis=1)
    Hp = lap_pinv(hess(beta, D))
    return Hp @ (Ud.T @ Ud) @ Hp


def build_bt_model(M):
    """A single-endpoint BT model for numpyro NUTS: beta ~ flat sum-to-zero
    prior via the Helmert basis (same identifiability trick as
    blam_analysis.build_model), wins[i,j] for i<j ~ Binomial(D, sigmoid(beta_i
    - beta_j)). Unlike `fit`, this has no analytic (co)variance -- its
    uncertainty is whatever the NUTS posterior says.
    """
    iu0, iu1 = (jnp.array(a) for a in np.triu_indices(M, 1))
    Q = jnp.array(helmert(M, full=False).T)

    def model(Wobs, D):
        z = numpyro.sample("z", dist.Normal(0.0, 10.0).expand([M - 1]).to_event(1))
        beta = numpyro.deterministic("beta", Q @ z)
        pi = jax.nn.sigmoid(beta[iu0] - beta[iu1])
        numpyro.sample("wins", dist.Binomial(total_count=D, probs=pi), obs=Wobs)

    return model


def simulate_bayes_batch(M, D, reps, rng, beta_true=None, coupled=True,
                          num_warmup=300, num_samples=500, seed0=0):
    """Monte Carlo batch of independent NUTS(numpyro) BT fits -- one fresh
    posterior per replicate (unlike HMC chains of the *same* posterior,
    there's nothing to combine across replicates during sampling). Returns
    raw per-replicate arrays rather than a summary so that
    scripts/modal_pl_variance_inflation.py can concatenate batches from many
    containers before computing the Monte Carlo covariance across every
    replicate at once (see `summarize_bayes`).
    """
    beta_true = np.zeros(M) if beta_true is None else beta_true
    iu = np.triu_indices(M, 1)
    model = build_bt_model(M)
    mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=False)

    hats = np.empty((reps, M))
    v_post = np.empty(reps)
    tr_post = np.empty(reps)
    for r in range(reps):
        W, _ = sample(beta_true, D, rng, coupled)
        Wobs = jnp.asarray(W[iu[0], iu[1]])
        mcmc.run(random.PRNGKey(seed0 + r), Wobs, D)
        post_beta = np.asarray(mcmc.get_samples()["beta"])  # (num_samples, M)
        hats[r] = post_beta.mean(axis=0)
        post_cov = np.cov(post_beta.T)
        v_post[r] = post_cov[0, 0] + post_cov[1, 1] - 2 * post_cov[0, 1]  # c @ post_cov @ c, c = e0 - e1
        tr_post[r] = np.trace(post_cov)
    return hats, v_post, tr_post


def summarize_bayes(hats, v_post, tr_post, beta_true):
    """Same summary as `run`'s return dict, but for `simulate_bayes_batch`'s
    posterior-based estimates -- there's no naive/sandwich split here, just
    the raw NUTS posterior, so `vif` is the direct analog of `run`'s
    vif_emp: how much the Monte Carlo variance of beta_hat exceeds what the
    average posterior already reports."""
    c = np.zeros(hats.shape[1])
    c[0], c[1] = 1.0, -1.0
    truth = c @ beta_true

    emp_cov = np.cov(hats.T)
    err = np.abs(hats @ c - truth)
    return {
        "emp": c @ emp_cov @ c,
        "post": v_post.mean(),
        "vif": np.trace(emp_cov) / tr_post.mean(),
        "cov_post": np.mean(err <= NOMINAL_Z * np.sqrt(v_post)),
        "bias": hats.mean(axis=0) @ c - truth,
    }


def report_bayes(tag, res, M, D, reps):
    print(f"\n{tag}  (M={M}, D={D}, reps={reps})")
    print(f"  Var(b0-b1) empirical      {res['emp']:.5f}")
    print(f"  Var(b0-b1) posterior      {res['post']:.5f}   <- mean NUTS posterior variance, no sandwich correction")
    print(f"  VIF empirical / theory    {res['vif']:.3f} / {(M + 1) / 3:.3f}")
    print(f"  95% credible-interval coverage   {res['cov_post']:.3f}")
    print(f"  bias of b0-b1             {res['bias']:+.4f}")


def run(M, D, reps, rng, beta_true=None, coupled=True):
    beta_true = np.zeros(M) if beta_true is None else beta_true
    c = np.zeros(M)
    c[0], c[1] = 1.0, -1.0
    truth = c @ beta_true

    hats = np.empty((reps, M))
    v_naive = np.empty(reps)
    v_sand = np.empty(reps)
    tr_naive = np.empty(reps)
    tr_sand = np.empty(reps)

    for r in range(reps):
        W, R = sample(beta_true, D, rng, coupled)
        b = fit(W, D)
        hats[r] = b
        Hp = lap_pinv(hess(b, D))
        S = sandwich(b, R, D)
        v_naive[r], v_sand[r] = c @ Hp @ c, c @ S @ c
        tr_naive[r], tr_sand[r] = np.trace(Hp), np.trace(S)

    emp_cov = np.cov(hats.T)
    err = np.abs(hats @ c - truth)
    return {
        "emp": c @ emp_cov @ c,
        "naive": v_naive.mean(),
        "sand": v_sand.mean(),
        "vif_emp": np.trace(emp_cov) / tr_naive.mean(),
        "vif_sand": tr_sand.mean() / tr_naive.mean(),
        "cov_naive": np.mean(err <= NOMINAL_Z * np.sqrt(v_naive)),
        "cov_sand": np.mean(err <= NOMINAL_Z * np.sqrt(v_sand)),
        "bias": hats.mean(axis=0) @ c - truth,
    }


def report(tag, res, M, D):
    print(f"\n{tag}  (M={M}, D={D})")
    print(f"  Var(b0-b1) empirical      {res['emp']:.5f}")
    print(f"  Var(b0-b1) model-based    {res['naive']:.5f}   <- H^+, what the BT/BBT fit reports")
    print(f"  Var(b0-b1) sandwich       {res['sand']:.5f}   <- H^+JH^+, clustered by dataset")
    print(f"  VIF empirical / theory    {res['vif_emp']:.3f} / {(M + 1) / 3:.3f}"
          f"      sandwich estimate {res['vif_sand']:.3f}")
    print(f"  95% coverage naive / sandwich   {res['cov_naive']:.3f} / {res['cov_sand']:.3f}")
    print(f"  bias of b0-b1             {res['bias']:+.4f}")


def plot_coverage(scenarios, output):
    """Grouped bar chart of naive vs. sandwich 95% CI coverage of beta0-beta1,
    across `scenarios` ((label, res) pairs from `run`) -- shows the naive
    BT/BBT variance undercovers regardless of whether beta_true is null, and
    the sandwich estimator restores nominal coverage in both cases."""
    # Deferred: matplotlib isn't installed on scripts/modal_pl_variance_inflation.py's
    # image, which only imports this module for its numpy/jax/numpyro functions.
    import matplotlib.pyplot as plt

    labels = [label for label, _ in scenarios]
    naive = [res["cov_naive"] for _, res in scenarios]
    sand = [res["cov_sand"] for _, res in scenarios]

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.bar(x - width / 2, naive, width, color=COLOR_NAIVE, label="Naive (H⁺)")
    ax.bar(x + width / 2, sand, width, color=COLOR_SANDWICH, label="Sandwich (H⁺JH⁺)")
    ax.axhline(0.95, color=COLOR_TARGET, linestyle="--", linewidth=1, zorder=0)
    ax.text(0.5, 1.02, "95% nominal", color=COLOR_TARGET, fontsize=8, ha="center")

    for xi, v in zip(x - width / 2, naive):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=COLOR_TEXT_PRIMARY)
    for xi, v in zip(x + width / 2, sand):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=COLOR_TEXT_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=COLOR_TEXT_PRIMARY, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("95% CI coverage of beta0-beta1", color=COLOR_TEXT_MUTED, fontsize=9)
    ax.tick_params(axis="y", colors=COLOR_TEXT_MUTED, labelsize=8)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_GRID)

    ax.set_title(
        "Naive BT variance undercovers; sandwich doesn't", loc="left",
        color=COLOR_TEXT_PRIMARY, fontsize=11, fontweight="bold", pad=10,
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2,
        frameon=False, fontsize=8, labelcolor=COLOR_TEXT_MUTED,
    )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nWrote {output}")


def plot_bayes_variance(scenarios, output):
    """Grouped bar chart of the NUTS posterior's reported variance of
    beta0-beta1 vs. its true (empirical, across replicates) variance, across
    `scenarios` ((label, res) pairs from `summarize_bayes`) -- shows the
    posterior underreports its own uncertainty by the same VIF as the naive
    Fisher-info estimator, whether or not beta_true is null."""
    import matplotlib.pyplot as plt

    labels = [label for label, _ in scenarios]
    reported = [res["post"] for _, res in scenarios]
    true = [res["emp"] for _, res in scenarios]

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.bar(x - width / 2, reported, width, color=COLOR_NAIVE, label="Reported (posterior)")
    ax.bar(x + width / 2, true, width, color=COLOR_SANDWICH, label="True (empirical)")

    for xi, v in zip(x - width / 2, reported):
        ax.text(xi, v + max(true) * 0.02, f"{v:.4f}", ha="center", fontsize=8, color=COLOR_TEXT_PRIMARY)
    for xi, v in zip(x + width / 2, true):
        ax.text(xi, v + max(true) * 0.02, f"{v:.4f}", ha="center", fontsize=8, color=COLOR_TEXT_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=COLOR_TEXT_PRIMARY, fontsize=9)
    ax.set_ylim(0, max(true) * 1.25)
    ax.set_ylabel("Var(beta0-beta1)", color=COLOR_TEXT_MUTED, fontsize=9)
    ax.tick_params(axis="y", colors=COLOR_TEXT_MUTED, labelsize=8)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_GRID)

    ax.set_title(
        "BBT posterior underreports its own variance", loc="left",
        color=COLOR_TEXT_PRIMARY, fontsize=11, fontweight="bold", pad=10,
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2,
        frameon=False, fontsize=8, labelcolor=COLOR_TEXT_MUTED,
    )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nWrote {output}")


def main():
    rng = np.random.default_rng(20260826)

    M, D, reps = 10, 200, 3000
    res_a = run(M, D, reps, rng)
    report("A. Plackett-Luce (one ranking per dataset)", res_a, M, D)
    print(f"   theory: empirical {8 * (M + 1) / (3 * M * D):.5f}, naive {8 / (M * D):.5f}")

    report("B. Control: pairs drawn independently",
           run(M, D, reps, rng, coupled=False), M, D)
    print("   theory: VIF = 1, both variances agree")

    beta_true = np.linspace(-1.5, 1.5, M)
    res_c = run(M, D, reps, rng, beta_true=beta_true)
    report("C. Plackett-Luce, non-null abilities", res_c, M, D)

    plot_coverage([("Null\n(beta = 0)", res_a), ("Non-null\n(beta != 0)", res_c)], OUTPUT_PNG)

    print("\nD. VIF vs M   (D=100)")
    print("   M    empirical   sandwich   (M+1)/3")
    for M_ in (2, 3, 5, 10, 20, 40):
        r = run(M_, 100, 800, rng)
        print(f"  {M_:3d}   {r['vif_emp']:8.3f}   {r['vif_sand']:8.3f}   {(M_ + 1) / 3:7.3f}")

    print("\nE. VIF vs D   (M=10, does not wash out with more datasets)")
    print("     D    empirical   sandwich   (M+1)/3")
    for D_ in (25, 50, 100, 200, 400):
        r = run(10, D_, 800, rng)
        print(f"  {D_:4d}   {r['vif_emp']:8.3f}   {r['vif_sand']:8.3f}   {11 / 3:7.3f}")

    # F: does the posterior itself (not just the naive Fisher-info variance)
    # inherit this inflation? The MLE's VIF carrying over to the Bayesian
    # model relies on Bernstein-von-Mises, so check directly: fit the same
    # pairwise BT likelihood with NUTS per replicate and see whether the
    # posterior's own credible-interval coverage undercovers like `naive`
    # did in A/C. Each replicate is a full NUTS fit (~1.3s locally after JIT
    # warmup), so this uses far fewer reps than A-E as a local smoke test --
    # scripts/modal_pl_variance_inflation.py fans it out across containers
    # at the full reps=3000 scale.
    print("\nF. HMC posterior (numpyro NUTS) -- local smoke test, reps << A-E")
    reps_bayes = 60
    hats_f, v_post_f, tr_post_f = simulate_bayes_batch(M, D, reps_bayes, rng)
    report_bayes("F. Plackett-Luce, NUTS posterior, null abilities",
                 summarize_bayes(hats_f, v_post_f, tr_post_f, np.zeros(M)), M, D, reps_bayes)


if __name__ == "__main__":
    main()
