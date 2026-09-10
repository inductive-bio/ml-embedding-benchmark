"""Bayesian latent-ability model (BLAM) for ranking embeddings across OpenADMET
endpoints, per docs/bayesian_model.md.

Pipeline:
  1. Per (endpoint, embedding), take the |error| of whichever head (rf/ridge/
     knn/tabicl/tabpfn) has the best train+valid CV score -- selecting by CV
     score rather than test error avoids a winner's-curse leak. An embedding
     is included only if every head finished for that endpoint.
  2. Union those per-embedding series into one compounds x embeddings error
     matrix A_e per endpoint (doc's Step 1; NaN where an embedding has no
     observation for a compound, since embed-time failures drop different
     compounds per embedding). A_e is the sufficient statistic -- y_hat_e
     (Step 2) and V_e (Step 3) are computed from it, and nothing downstream
     touches per-compound data again.
  3. Fit the hierarchical model from the doc via NUTS (numpyro): global latent
     ability theta (sum-to-zero across all embeddings), per-endpoint
     difficulty mu_e and discrimination beta_e, and per-endpoint specialization
     scale omega_e.
  4. u_ie (specialization) is never sampled directly -- Step 4 of the doc
     marginalizes it into omega_e^2*I + V_e. It's recovered afterwards by
     drawing from its posterior conditional given r_e = y_hat_e - mu_e*1 +
     beta_e*theta_e, one draw per posterior sample.

Usage:
    python scripts/blam_analysis.py
    python scripts/blam_analysis.py --errors-dir data/errors --output data/blam
"""

import argparse
import glob
import logging as log
import os

import jax
import jax.numpy as jnp
import jax.random as random
import joblib
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS
from scipy.linalg import helmert

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_METHODS = ("rf", "ridge", "knn", "tabicl", "tabpfn")
# Floor for the noise-corrected sample variance (docs/bayesian_model.md
# "Priors" section) -- keeps sigma_beta/sigma_omega's LogNormal location
# finite if measurement noise alone would explain all of s_e^2.
EPSILON_0 = 1e-6


def discover_endpoint_dirs(errors_dir: str, dataset_glob: str) -> list:
    return sorted(p for p in glob.glob(os.path.join(errors_dir, dataset_glob)) if os.path.isdir(p))


def discover_embeddings(endpoint_dir: str, methods: tuple, allowed: frozenset = None) -> list:
    stems = set()
    for path in glob.glob(os.path.join(endpoint_dir, "*.joblib")):
        name = os.path.splitext(os.path.basename(path))[0]
        if "__" not in name:
            continue
        embedding, method = name.rsplit("__", 1)
        if method in methods and (allowed is None or embedding in allowed):
            stems.add(embedding)
    return sorted(stems)


def load_best_head_error(endpoint_dir: str, embedding: str, methods: tuple):
    """|error| of the best-by-cv_score head across `methods` for one
    (embedding, endpoint), indexed by molecule_id. Returns None if any method
    is missing -- see module docstring on why partial coverage isn't allowed
    (every embedding must pick from the same pool of heads)."""
    per_method = {}
    cv_scores = {}
    for method in methods:
        path = os.path.join(endpoint_dir, f"{embedding}__{method}.joblib")
        if not os.path.exists(path):
            return None
        result = joblib.load(path)
        series = pd.Series(np.abs(result["error"]), index=result["molecule_id"], name=method)
        if series.index.has_duplicates:
            raise ValueError(f"duplicate molecule_id in {path}")
        per_method[method] = series
        cv_scores[method] = result["cv_score"]
    combined = pd.concat(per_method.values(), axis=1)
    if combined.isna().any().any():
        # All heads are fit on the same embedding-derived train/test split
        # (compute_prediction_errors.py), so a NaN here means the method
        # joblibs disagree on which compounds were held out -- a data
        # integrity problem, not benign missingness.
        raise ValueError(f"'{embedding}' at '{endpoint_dir}': methods disagree on test-set molecule_ids")
    best_method = max(cv_scores, key=cv_scores.get)
    return per_method[best_method]


def load_finetuned_error(finetune_dir: str, endpoint_name: str, file_stem: str):
    """|error| of an end-to-end fine-tuned model's (untagged) result for one
    (model, endpoint), indexed by molecule_id. Returns None if that model
    wasn't fine-tuned for this endpoint. Deliberately reads only
    '{file_stem}__finetuned.joblib' (no '_<tag>' suffix) -- tagged files are
    hyperparameter-sweep pilots, not the model's canonical result."""
    path = os.path.join(finetune_dir, endpoint_name, f"{file_stem}__finetuned.joblib")
    if not os.path.exists(path):
        return None
    result = joblib.load(path)
    series = pd.Series(np.abs(result["error"]), index=result["molecule_id"], name=file_stem)
    if series.index.has_duplicates:
        raise ValueError(f"duplicate molecule_id in {path}")
    return series


def build_error_matrix(
    endpoint_dir: str,
    methods: tuple,
    allowed: frozenset = None,
    finetune_dir: str = None,
    finetuned_specs: list = None,
):
    """The endpoint's A_e: compounds x embeddings, unioned across embeddings'
    own test compounds (NaN = that embedding never observed that compound).

    finetuned_specs (entity_name, file_stem) pairs -- e.g. ("unimolv1_finetuned",
    "unimolv1") -- are added as additional columns sourced from
    finetune_dir/<endpoint>/<file_stem>__finetuned.joblib instead of
    data/errors' rf/ridge/knn/tabicl/tabpfn heads. End-to-end fine-tuning and
    frozen-embedding-plus-shallow-head are different strategies, not
    competing "heads" for the same embedding, so a model with both gets two
    separate entities here rather than one replacing the other.
    """
    columns = {}
    for embedding in discover_embeddings(endpoint_dir, methods, allowed):
        series = load_best_head_error(endpoint_dir, embedding, methods)
        if series is not None:
            columns[embedding] = series
    if finetune_dir and finetuned_specs:
        endpoint_name = os.path.basename(endpoint_dir)
        for entity_name, file_stem in finetuned_specs:
            series = load_finetuned_error(finetune_dir, endpoint_name, file_stem)
            if series is not None:
                columns[entity_name] = series.rename(entity_name)
    return pd.DataFrame(columns) if columns else None


def ensure_psd(V: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Pairwise-deletion covariance estimates (unlike full-sample covariance)
    aren't guaranteed PSD -- each entry is computed from a different subset of
    shared compounds. Clip negative/near-zero eigenvalues before using V as a
    Gaussian covariance block."""
    eigvals, eigvecs = np.linalg.eigh(V)
    if eigvals.min() >= floor:
        return V
    eigvals = np.clip(eigvals, floor, None)
    V_psd = (eigvecs * eigvals) @ eigvecs.T
    return (V_psd + V_psd.T) / 2


def compute_sampling_stats(A: pd.DataFrame):
    """Steps 2-3 of docs/bayesian_model.md, generalized to a unioned (not
    intersected) error matrix: y_hat_e (per-embedding mean) and V_e (sampling
    covariance of those means), using pairwise-complete compounds for each
    (i, j) entry since embeddings don't all share the same compound set."""
    X = A.to_numpy()
    n_compounds, n_embeddings = X.shape
    y_hat = np.nanmean(X, axis=0)
    V = np.empty((n_embeddings, n_embeddings))
    n_pairs = np.empty((n_embeddings, n_embeddings), dtype=int)
    for i in range(n_embeddings):
        xi = X[:, i]
        for j in range(i, n_embeddings):
            xj = X[:, j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            n_ij = int(mask.sum())
            if n_ij < 2:
                raise ValueError(
                    f"embeddings '{A.columns[i]}' and '{A.columns[j]}' share fewer than 2 "
                    "compounds -- can't estimate their sampling covariance"
                )
            cov = np.sum((xi[mask] - y_hat[i]) * (xj[mask] - y_hat[j])) / (n_ij - 1)
            V[i, j] = V[j, i] = cov / n_ij
            n_pairs[i, j] = n_pairs[j, i] = n_ij
    return y_hat, ensure_psd(V), n_pairs


class EndpointStat:
    def __init__(self, name, embeddings, y_hat, V, n_compounds):
        self.name = name
        self.embeddings = embeddings  # column order matching y_hat/V
        self.y_hat = y_hat
        self.V = V
        self.n_compounds = n_compounds  # diagnostic only (union size)


def load_endpoint_stats(
    errors_dir: str,
    dataset_glob: str,
    methods: tuple,
    allowed: frozenset = None,
    finetune_dir: str = None,
    finetuned_specs: list = None,
) -> list:
    stats = []
    for endpoint_dir in discover_endpoint_dirs(errors_dir, dataset_glob):
        name = os.path.basename(endpoint_dir)
        A = build_error_matrix(endpoint_dir, methods, allowed, finetune_dir, finetuned_specs)
        if A is None:
            log.warning(f"'{name}': no embedding has all of {methods} -- skipping endpoint")
            continue
        if A.shape[1] < 2:
            # A single embedding can't identify beta_e/theta separately from
            # mu_e for this endpoint -- it contributes nothing to the model.
            log.warning(f"'{name}': only {A.shape[1]} embedding(s) with full method coverage -- skipping endpoint")
            continue
        y_hat, V, n_pairs = compute_sampling_stats(A)
        log.info(
            f"'{name}': {A.shape[1]} embeddings, {A.shape[0]} unioned compounds "
            f"(pairwise overlap {n_pairs.min()}-{n_pairs.max()})"
        )
        stats.append(EndpointStat(name, list(A.columns), y_hat, V, A.shape[0]))
    return stats


def compute_prior_scale(stats: list) -> float:
    """s from the doc's Priors section: a noise-corrected, cross-endpoint
    median spread of log-MAE, used as the LogNormal location for
    sigma_beta/sigma_omega."""
    s_tilde = []
    for stat in stats:
        s_e2 = np.var(stat.y_hat, ddof=1)
        diag = np.diag(stat.V)
        off_diag_sum = stat.V.sum() - diag.sum()
        n = len(stat.embeddings)
        v_bar = diag.mean()
        c_bar = off_diag_sum / (n * n - n)
        s_tilde.append(np.sqrt(max(s_e2 - (v_bar - c_bar), EPSILON_0)))
    return float(np.median(s_tilde))


def build_model(global_embeddings: list, stats: list, prior_log_s: float):
    n_embeddings = len(global_embeddings)
    emb_index = {name: i for i, name in enumerate(global_embeddings)}
    # theta ~ N(0, I - 11^T/M) via theta = Q @ z, z ~ N(0, I_{M-1}) -- Helmert
    # rows are an orthonormal basis for the subspace orthogonal to 1, so this
    # gives the doc's sum-to-zero identifiability constraint for free.
    Q = jnp.array(helmert(n_embeddings, full=False).T)

    def model():
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n_embeddings - 1]).to_event(1))
        theta = numpyro.deterministic("theta", Q @ z)

        sigma_beta = numpyro.sample("sigma_beta", dist.LogNormal(prior_log_s, 0.5))
        sigma_omega = numpyro.sample("sigma_omega", dist.LogNormal(prior_log_s, 0.5))

        for stat in stats:
            idx = jnp.array([emb_index[e] for e in stat.embeddings])
            mu_e = numpyro.sample(f"mu__{stat.name}", dist.Normal(float(np.mean(stat.y_hat)), 1.0))
            beta_tilde = numpyro.sample(f"beta_tilde__{stat.name}", dist.HalfNormal(1.0))
            omega_tilde = numpyro.sample(f"omega_tilde__{stat.name}", dist.HalfNormal(1.0))
            beta_e = numpyro.deterministic(f"beta__{stat.name}", sigma_beta * beta_tilde)
            omega_e = numpyro.deterministic(f"omega__{stat.name}", sigma_omega * omega_tilde)

            theta_e = theta[idx]
            mean_e = mu_e - beta_e * theta_e
            cov_e = omega_e**2 * jnp.eye(len(stat.embeddings)) + jnp.array(stat.V)
            numpyro.sample(
                f"y__{stat.name}",
                dist.MultivariateNormal(mean_e, covariance_matrix=cov_e),
                obs=jnp.array(stat.y_hat),
            )

    return model


def run_mcmc(model, seed: int, num_warmup: int, num_samples: int, num_chains: int):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(random.PRNGKey(seed))
    mcmc.print_summary(exclude_deterministic=False)
    return mcmc.get_samples()


def run_single_chain(model, seed: int, num_warmup: int, num_samples: int, target_accept_prob: float = 0.8) -> dict:
    """One independent NUTS chain. HMC chains are embarrassingly parallel, so
    scripts/modal_blam_analysis.py fans this out across separate Modal
    containers (each with its own dedicated CPU) rather than relying on
    MCMC(..., num_chains=N)'s in-process chain parallelism, which would
    contend for one container's cores.

    target_accept_prob defaults to NUTS's own default (0.8), sufficient for
    this module's model. blam_robustness_checks.py's equicorrelated extension
    needs 0.9: its extra xi_i factor sits in a near-degenerate ridge with
    theta_i whenever kappa_e=a_e/omega_e is similar across endpoints, and 0.8
    leaves chains stuck in different corners of that ridge (R-hat up to ~6.8)
    even though each chain looks internally well-mixed.
    """
    kernel = NUTS(model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=False)
    mcmc.run(random.PRNGKey(seed))
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}


def combine_chains(chain_samples: list) -> tuple:
    """Combine independently-run chains' post-warmup draws into (flat_samples,
    grouped_samples). flat_samples concatenates every chain's draws into one
    (total_draws, ...) array per site, matching what mcmc.get_samples()
    returns for an in-process multi-chain run. grouped_samples stacks chains
    as a leading axis (num_chains, num_samples, ...), the shape
    numpyro.diagnostics.summary() needs for r_hat/n_eff across chains that
    never shared a process.
    """
    flat_samples = {k: np.concatenate([c[k] for c in chain_samples], axis=0) for k in chain_samples[0]}
    grouped_samples = {k: np.stack([c[k] for c in chain_samples], axis=0) for k in chain_samples[0]}
    return flat_samples, grouped_samples


def compute_specialization(samples: dict, stats: list, seed: int = 0) -> dict:
    """Draws u_{.e}^(s) from its posterior conditional given r_{.e}^(s)
    (docs/bayesian_model.md Step 4/5), one draw per posterior sample -- not
    just the conditional mean.

    (u_{.e}, r_{.e}) are jointly Gaussian with Cov(u, r) = omega_e^2*I and
    Var(r) = Sigma_e = omega_e^2*I + V_e, so
        u_{.e} | r_{.e} ~ N(omega_e^2 Sigma_e^-1 r_{.e},
                            omega_e^2*I - omega_e^4 Sigma_e^-1).
    Using only the conditional mean (the BLUP) would understate u's true
    posterior spread.

    Sigma_e depends on each draw only through the scalar omega_e, so both
    moments diagonalize in V_e's fixed eigenbasis (V_e = Q Lambda Q^T,
    computed once per endpoint):
        mean eigenvalue = omega^2 / (lambda + omega^2)
        cov eigenvalue  = omega^2 * lambda / (lambda + omega^2)
    both always in [0, omega^2], so no jitter/Cholesky is needed beyond
    clipping Lambda itself."""
    rng = np.random.default_rng(seed)
    u_by_endpoint = {}
    for stat in stats:
        mu_e = np.asarray(samples[f"mu__{stat.name}"])  # (S,)
        beta_e = np.asarray(samples[f"beta__{stat.name}"])  # (S,)
        omega_e = np.asarray(samples[f"omega__{stat.name}"])  # (S,)
        theta_e = stat.theta_samples  # (S, n_e), attached by caller

        r_e = stat.y_hat[None, :] - mu_e[:, None] + beta_e[:, None] * theta_e  # (S, n_e)

        lam, Q = np.linalg.eigh(stat.V)  # V_e = Q @ diag(lam) @ Q.T
        lam = np.clip(lam, 0.0, None)

        w2 = omega_e[:, None] ** 2  # (S, 1)
        denom = lam[None, :] + w2  # (S, n_e)
        r_rot = r_e @ Q  # (S, n_e)
        mean_rot = w2 * r_rot / denom
        sd_rot = np.sqrt(w2 * lam[None, :] / denom)
        z = rng.standard_normal(mean_rot.shape)
        u_by_endpoint[stat.name] = (mean_rot + sd_rot * z) @ Q.T
    return u_by_endpoint


def summarize(global_embeddings: list, samples: dict, stats: list, u_by_endpoint: dict):
    theta = np.asarray(samples["theta"])
    theta_summary = pd.DataFrame({
        "embedding": global_embeddings,
        "theta_mean": theta.mean(axis=0),
        "theta_std": theta.std(axis=0),
    }).sort_values("theta_mean", ascending=False).reset_index(drop=True)

    rows = []
    for stat in stats:
        u_e = u_by_endpoint[stat.name]
        for j, embedding in enumerate(stat.embeddings):
            rows.append({
                "endpoint": stat.name,
                "embedding": embedding,
                "u_mean": u_e[:, j].mean(),
                "u_std": u_e[:, j].std(),
            })
    specialization_summary = pd.DataFrame(rows)

    endpoint_rows = []
    for stat in stats:
        mu = np.asarray(samples[f"mu__{stat.name}"])
        beta = np.asarray(samples[f"beta__{stat.name}"])
        omega = np.asarray(samples[f"omega__{stat.name}"])
        endpoint_rows.append({
            "endpoint": stat.name,
            "mu_mean": mu.mean(),
            "mu_std": mu.std(),
            "beta_mean": beta.mean(),
            "beta_std": beta.std(),
            "omega_mean": omega.mean(),
            "omega_std": omega.std(),
            "n_compounds": stat.n_compounds,
            "n_embeddings": len(stat.embeddings),
        })
    endpoint_summary = pd.DataFrame(endpoint_rows)

    return theta_summary, specialization_summary, endpoint_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--errors-dir", default="data/errors")
    parser.add_argument("--dataset-glob", default="reg_openadmet_*", help="Glob over endpoint dirs under --errors-dir")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument(
        "--embeddings",
        default=None,
        help="Comma-separated embedding stems to include (default: all discovered). "
        "Use to drop duplicate variants (e.g. keep only the best ChemBERTa checkpoint).",
    )
    parser.add_argument(
        "--embeddings-file",
        default=None,
        help="Path to a file with one embedding stem per line, as an alternative to --embeddings.",
    )
    parser.add_argument("--finetune-errors-dir", default="data/finetune_errors")
    parser.add_argument(
        "--finetuned-models",
        default=None,
        help="Comma-separated end-to-end fine-tuned model stems to add as extra "
        "'<name>_finetuned' entities (alongside, not instead of, any same-named "
        "frozen embedding). Each entry is 'EntityName' or 'EntityName=file_stem' "
        "when the finetune_errors filename differs from the entity name (e.g. "
        "'SimSon=simson' for a casing mismatch).",
    )
    parser.add_argument(
        "--finetuned-models-file",
        default=None,
        help="Path to a file with one --finetuned-models entry per line, as an alternative to --finetuned-models.",
    )
    parser.add_argument("--num-warmup", type=int, default=2000)
    parser.add_argument("--num-samples", type=int, default=4000)
    parser.add_argument("--num-chains", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--specialization-seed", type=int, default=0, help="Seed for compute_specialization's conditional draw of u_ie"
    )
    parser.add_argument("--output", default="data/blam")
    args = parser.parse_args()

    numpyro.set_host_device_count(args.num_chains)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    allowed = None
    if args.embeddings_file:
        with open(args.embeddings_file) as f:
            allowed = frozenset(line.strip() for line in f if line.strip())
    elif args.embeddings:
        allowed = frozenset(e.strip() for e in args.embeddings.split(",") if e.strip())
    if allowed is not None:
        log.info(f"Restricting to {len(allowed)} embedding(s) from --embeddings/--embeddings-file")

    finetuned_entries = None
    if args.finetuned_models_file:
        with open(args.finetuned_models_file) as f:
            finetuned_entries = [line.strip() for line in f if line.strip()]
    elif args.finetuned_models:
        finetuned_entries = [e.strip() for e in args.finetuned_models.split(",") if e.strip()]
    finetuned_specs = None
    if finetuned_entries:
        finetuned_specs = []
        for entry in finetuned_entries:
            if "=" in entry:
                entity, file_stem = entry.split("=", 1)
            else:
                entity, file_stem = entry, entry
            finetuned_specs.append((f"{entity}_finetuned", file_stem))
        log.info(f"Adding {len(finetuned_specs)} fine-tuned entity(ies): {[e for e, _ in finetuned_specs]}")

    stats = load_endpoint_stats(
        args.errors_dir, args.dataset_glob, methods, allowed, args.finetune_errors_dir, finetuned_specs
    )
    if not stats:
        raise SystemExit("No endpoints had >=2 embeddings with full method coverage -- nothing to fit")

    global_embeddings = sorted({e for stat in stats for e in stat.embeddings})
    log.info(f"Fitting {len(global_embeddings)} embeddings across {len(stats)} endpoints")

    prior_log_s = np.log(compute_prior_scale(stats) / np.sqrt(2))
    model = build_model(global_embeddings, stats, prior_log_s)
    samples = run_mcmc(model, args.seed, args.num_warmup, args.num_samples, args.num_chains)

    emb_index = {name: i for i, name in enumerate(global_embeddings)}
    theta = np.asarray(samples["theta"])
    for stat in stats:
        idx = [emb_index[e] for e in stat.embeddings]
        stat.theta_samples = theta[:, idx]

    u_by_endpoint = compute_specialization(samples, stats, args.specialization_seed)
    theta_summary, specialization_summary, endpoint_summary = summarize(global_embeddings, samples, stats, u_by_endpoint)

    os.makedirs(args.output, exist_ok=True)
    joblib.dump(
        {"samples": samples, "stats": stats, "u_by_endpoint": u_by_endpoint, "global_embeddings": global_embeddings},
        os.path.join(args.output, "posterior.joblib"),
    )
    theta_summary.to_csv(os.path.join(args.output, "theta_summary.csv"), index=False)
    specialization_summary.to_csv(os.path.join(args.output, "specialization_summary.csv"), index=False)
    endpoint_summary.to_csv(os.path.join(args.output, "endpoint_summary.csv"), index=False)

    log.info(f"Wrote results to {args.output}/")
    print("\nLatent ability ranking (theta, higher = better):")
    print(theta_summary.to_string(index=False))


if __name__ == "__main__":
    main()
