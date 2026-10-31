"""Six robustness checks for the BLAM fit copied into data/blam_tests (raw
copy of data/blam_v2_monroe's posterior.joblib etc., see data/blam_tests/raw).

  1. truncation:       drop the low-theta embeddings, refit, compare
                        per-endpoint P(i > ECFP) for the retained embeddings.
  2. heteroscedastic:  |r_ie| vs y_hat_ie per endpoint -- free, no refit.
  3. correlation:      empirical vs BLAM-implied endpoint correlation of
                        y_hat_.e, effective number of independent endpoints.
  4. loglink:          refit on log(y_hat_ie) with V_e delta-mapped by
                        D^-1 V_e D^-1, D = diag(y_hat_.e); Kendall's tau vs
                        the current theta ranking.
  5. endpoint_psi:      Psi_hat, the cross-endpoint correlation
                        (correlation of the standardized residual z_ie),
                        lambda_1(Psi_hat)/VIF/E_eff/sqrt(VIF), max |Psi_hat_ee'|
                        overall and within a priori related endpoint blocks,
                        and a PPC p-value calibrating lambda_1 against
                        replicates simulated under Psi=I and refit through
                        the identical pipeline.
  6. endpoint_psi_sensitivity: refit with an equicorrelated Psi at rho in
                        {0, 0.25, 0.5, 0.75} (u_ie = omega_e*(sqrt(1-rho)*eps_ie
                        + sqrt(rho)*xi_i), a common per-model factor xi_i
                        shared across endpoints), reporting P(Monroe>ECFP) on
                        a new endpoint and the Spearman rank correlation of
                        theta against the rho=0 (Table 2) ranking at each
                        rung, plus the closed-form VIF(rho)=(1-rho)+rho*wbar
                        rho->1 bound (no refit needed for the bound).

Also builds a "MeanPredictor" 49th entity (mean-of-train-set-style null
model, |y_true - endpoint mean| per compound) folded into the truncation
refit, and -- only if a test fires -- an omega-heteroscedasticity extension
log(omega_ie) = log(omega_e) - gamma*theta_i, nesting the current model at
gamma=0.

Usage:
    python scripts/blam_robustness_checks.py <truncation|heteroscedastic|correlation|loglink|omega_extension|endpoint_psi|endpoint_psi_sensitivity> [--num-warmup N --num-samples N --num-chains N]
"""

import argparse
import logging as log
import os
import sys

import joblib
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS
from scipy.special import ndtr
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from blam_analysis import (  # noqa: E402
    DEFAULT_METHODS,
    EPSILON_0,
    EndpointStat,
    build_error_matrix,
    build_model,
    combine_chains,
    compute_prior_scale,
    compute_sampling_stats,
    compute_specialization,
    discover_endpoint_dirs,
    ensure_psd,
    run_single_chain,
    summarize,
)

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

RAW_DIR = "data/blam_tests/raw"
OUT_DIR = "data/blam_tests"
ERRORS_DIR = "data/errors/errors"  # nested dir is the current one -- see investigation notes in PR description
FINETUNE_DIR = "data/finetune_errors"
DATASET_GLOB = "reg_openadmet_*"
BASELINE = "ECFP_count"
MONROE = "monroe"
THETA_CUTOFF = -1.0
FINETUNE_STEM_OVERRIDES = {"SimSon": "simson"}  # casing mismatch, see blam_analysis.py docstring
RHO_GRID = (0.0, 0.25, 0.5, 0.75)
# A priori related endpoint pairs (shared assay design/species), reused by
# both test3 (correlation) and test5 (endpoint_psi).
ENDPOINT_BLOCKS = {
    "Caco pair": ["reg_openadmet_expansionrx_Log_Caco_ER", "reg_openadmet_expansionrx_Log_Caco_Papp_AB"],
    "LogD pair": ["reg_openadmet_expansionrx_LogD", "reg_openadmet_polaris_admet_LogD"],
    "protein-binding triple": [
        "reg_openadmet_expansionrx_Log_Mouse_BPB", "reg_openadmet_expansionrx_Log_Mouse_MPB",
        "reg_openadmet_expansionrx_Log_Mouse_PPB",
    ],
}


def load_posterior(path=os.path.join(RAW_DIR, "posterior.joblib")):
    return joblib.load(path)


def finetuned_specs_for(global_embeddings):
    specs = []
    for e in global_embeddings:
        if e.endswith("_finetuned"):
            entity = e[: -len("_finetuned")]
            specs.append((e, FINETUNE_STEM_OVERRIDES.get(entity, entity)))
    return specs


def reload_error_matrices(global_embeddings, dataset_glob=DATASET_GLOB):
    """Rebuild each endpoint's raw compound x embedding |error| matrix A_e
    straight from data/errors + data/finetune_errors, matching the monroe
    run's 48-embedding set (validated against the stored y_hat in the caller)."""
    non_finetuned = frozenset(e for e in global_embeddings if not e.endswith("_finetuned"))
    specs = finetuned_specs_for(global_embeddings)
    mats = {}
    for endpoint_dir in discover_endpoint_dirs(ERRORS_DIR, dataset_glob):
        name = os.path.basename(endpoint_dir)
        A = build_error_matrix(endpoint_dir, DEFAULT_METHODS, non_finetuned, FINETUNE_DIR, specs)
        if A is not None:
            mats[name] = A
    return mats


def load_y_true(endpoint_dir: str) -> pd.Series:
    """Ground-truth target per molecule_id for one endpoint, merged across
    every (embedding, method) joblib in that dir -- y_true is a property of
    the compound/endpoint, not the model, so any file that touched a given
    molecule_id agrees on its value (asserted below)."""
    import glob

    merged = {}
    for path in glob.glob(os.path.join(endpoint_dir, "*.joblib")):
        result = joblib.load(path)
        for mid, val in zip(result["molecule_id"], result["y_true"]):
            if mid in merged and not np.isclose(merged[mid], val):
                raise ValueError(f"y_true mismatch for {mid} in {endpoint_dir}")
            merged[mid] = val
    return pd.Series(merged)


def add_mean_predictor(A: pd.DataFrame, endpoint_dir: str) -> pd.DataFrame:
    """49th entity: |y_true_c - mean(y_true over A's own compound union)|,
    the null-model ceiling -- an embedding with zero signal should land near
    this MAE. Uses A's own (union) compound set as the reference population
    so it's evaluated on the same compounds as everything else here."""
    y_true = load_y_true(endpoint_dir).reindex(A.index)
    mean_pred_error = (y_true - y_true.mean()).abs()
    out = A.copy()
    out["MeanPredictor"] = mean_pred_error
    return out


def endpoint_stats_from_matrices(mats: dict, endpoint_dirs: dict) -> list:
    stats = []
    for name, A in mats.items():
        A = add_mean_predictor(A, endpoint_dirs[name])
        y_hat, V, _ = compute_sampling_stats(A)
        stats.append(EndpointStat(name, list(A.columns), y_hat, V, A.shape[0]))
    return stats


def filter_outlier_chains(chain_samples: list, site: str = "theta", min_median_corr: float = 0.5) -> tuple:
    """Drop chains whose per-chain posterior mean (for `site`) doesn't
    correlate with the majority of the other chains.

    build_correlated_model's xi_i sits in a near-degenerate ridge with theta_i
    whenever kappa_e=a_e/omega_e is similar across endpoints (see
    run_single_chain's docstring). Raising target_accept_prob to 0.9 makes
    most chains find the same dominant region, but a growing minority (0, 2,
    2, 6 of 16 chains at rho=0, .25, .5, .75) still lands in a different,
    much-smaller-mass region and stays there (each such chain is internally
    well-mixed -- 0 divergences -- just stuck on the wrong side of the
    ridge). Flag and drop these by pairwise correlation of chain-mean theta
    vectors rather than letting them blow up R-hat for everyone."""
    means = [np.asarray(c[site]).mean(axis=0) for c in chain_samples]
    n = len(means)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = np.corrcoef(means[i], means[j])[0, 1]
            corr[i, j] = corr[j, i] = c
    median_corr = np.array([np.median(np.delete(corr[i], i)) for i in range(n)])
    keep = [i for i in range(n) if median_corr[i] >= min_median_corr]
    dropped = [i for i in range(n) if i not in keep]
    return [chain_samples[i] for i in keep], dropped


def run_chains(model, num_warmup, num_samples, num_chains, seed=0, target_accept_prob=0.8, filter_outliers=False):
    chain_samples = [
        run_single_chain(model, seed + i, num_warmup, num_samples, target_accept_prob) for i in range(num_chains)
    ]
    if filter_outliers and num_chains >= 4:
        chain_samples, dropped = filter_outlier_chains(chain_samples)
        if dropped:
            log.warning(
                f"Dropped {len(dropped)}/{num_chains} outlier chain(s) (indices {dropped}) "
                "that landed in a different posterior region -- see filter_outlier_chains docstring"
            )
    flat, grouped = combine_chains(chain_samples)
    return flat, grouped


def endpoint_ecfp_comparison(stats, samples, u_by_endpoint, baseline=BASELINE):
    """Same construction as scripts/visualize_blam.py's
    compute_endpoint_ecfp_comparison -- inlined so this script has no import
    dependency on matplotlib (visualize_blam.py's only other import)."""
    rows = []
    for stat in stats:
        if baseline not in stat.embeddings:
            continue
        beta_e = np.asarray(samples[f"beta__{stat.name}"])
        u_e = u_by_endpoint[stat.name]
        baseline_col = stat.embeddings.index(baseline)
        theta_baseline = stat.theta_samples[:, baseline_col]
        u_baseline = u_e[:, baseline_col]
        for col, embedding in enumerate(stat.embeddings):
            if embedding == baseline:
                continue
            d = beta_e * (stat.theta_samples[:, col] - theta_baseline) + (u_baseline - u_e[:, col])
            rows.append({
                "endpoint": stat.name,
                "embedding": embedding,
                "prob_beat_baseline": float((d > 0).mean()),
            })
    return pd.DataFrame(rows)


def fit_and_summarize(stats, global_embeddings, num_warmup, num_samples, num_chains, seed=0, spec_seed=0):
    prior_log_s = float(np.log(compute_prior_scale(stats) / np.sqrt(2)))
    model = build_model(global_embeddings, stats, prior_log_s)
    flat, grouped = run_chains(model, num_warmup, num_samples, num_chains, seed)

    emb_index = {name: i for i, name in enumerate(global_embeddings)}
    theta = np.asarray(flat["theta"])
    for stat in stats:
        idx = [emb_index[e] for e in stat.embeddings]
        stat.theta_samples = theta[:, idx]

    u_by_endpoint = compute_specialization(flat, stats, spec_seed)
    theta_summary, specialization_summary, endpoint_summary = summarize(global_embeddings, flat, stats, u_by_endpoint)
    return {
        "samples": flat, "grouped_samples": grouped, "stats": stats,
        "u_by_endpoint": u_by_endpoint, "global_embeddings": global_embeddings,
        "theta_summary": theta_summary, "specialization_summary": specialization_summary,
        "endpoint_summary": endpoint_summary,
    }


# ---------------------------------------------------------------------------
# Test 1: truncation
# ---------------------------------------------------------------------------

def cmd_truncation(args):
    posterior = load_posterior()
    global_embeddings = posterior["global_embeddings"]
    theta_mean = np.asarray(posterior["samples"]["theta"]).mean(axis=0)
    drop = sorted(e for e, t in zip(global_embeddings, theta_mean) if t < THETA_CUTOFF)
    log.info(f"Dropping {len(drop)} embeddings with theta_mean < {THETA_CUTOFF}: {drop}")

    log.info("Rebuilding raw error matrices from data/errors + data/finetune_errors...")
    mats = reload_error_matrices(global_embeddings)
    endpoint_dirs = {os.path.basename(d): d for d in discover_endpoint_dirs(ERRORS_DIR, DATASET_GLOB)}

    # data/ is gitignored, and some finetune_errors joblibs the original fit
    # used have since been deleted or regenerated with different content --
    # e.g. kermt_finetuned exists for only 8/17 endpoints today (though every
    # stored EndpointStat has it), and where it exists its values may no
    # longer match the stored fit. Exclude those columns entirely so the
    # embedding set stays identical across endpoints, and report this
    # separately since it's a data-availability issue, not the theta<cutoff
    # hypothesis under test.
    stat_by_name = {s.name: s for s in posterior["stats"]}
    missing_somewhere = set(global_embeddings) - set.intersection(*(set(A.columns) for A in mats.values()))
    stale = set()
    for name, A in mats.items():
        orig = stat_by_name[name]
        stored = pd.Series(orig.y_hat, index=orig.embeddings)
        for e in orig.embeddings:
            if e in A.columns and abs(float(A[e].mean()) - float(stored[e])) > 1e-6:
                stale.add(e)
    unavailable = sorted(missing_somewhere | stale)
    if unavailable:
        log.warning(f"Excluding {len(unavailable)} embedding(s) with missing/stale raw data on disk "
                    f"(missing somewhere: {sorted(missing_somewhere)}, stale values: {sorted(stale)})")

    retained = [e for e in global_embeddings if e not in drop and e not in unavailable]

    # Sanity check: reconstructed y_hat must match the stored fit's y_hat closely,
    # over the embeddings both sides have and that aren't already known-excluded.
    max_err = 0.0
    for name, A in mats.items():
        orig = stat_by_name[name]
        common = [e for e in orig.embeddings if e in A.columns and e not in unavailable]
        y_hat = A[common].mean(axis=0)
        err = float((y_hat - pd.Series(orig.y_hat, index=orig.embeddings)[common]).abs().max())
        max_err = max(max_err, err)
    log.info(f"Reconstruction check: max |y_hat_reconstructed - y_hat_stored| across all endpoints "
              f"(excluding known missing/stale columns) = {max_err:.2e}")
    if max_err > 1e-6:
        raise RuntimeError("Reconstructed error matrices do not match the stored fit -- wrong errors-dir/finetune specs")

    # Original fit's P(i > ECFP), restricted to the retained embeddings only.
    orig_stats_retained = []
    for s in posterior["stats"]:
        idx = [j for j, e in enumerate(s.embeddings) if e in retained]
        orig_stats_retained.append(EndpointStat(
            s.name, [s.embeddings[j] for j in idx], s.y_hat[idx], s.V[np.ix_(idx, idx)], s.n_compounds,
        ))
    theta_full = np.asarray(posterior["samples"]["theta"])
    emb_index_full = {e: i for i, e in enumerate(global_embeddings)}
    for s in orig_stats_retained:
        s.theta_samples = theta_full[:, [emb_index_full[e] for e in s.embeddings]]
    before = endpoint_ecfp_comparison(orig_stats_retained, posterior["samples"], posterior["u_by_endpoint"])

    # Refit on the retained + MeanPredictor set.
    retained_stats = endpoint_stats_from_matrices(
        {n: A[retained] for n, A in mats.items()}, endpoint_dirs,
    )
    retained_global = sorted({e for s in retained_stats for e in s.embeddings})
    log.info(f"Refitting {len(retained_global)} embeddings (incl. MeanPredictor) across {len(retained_stats)} endpoints")
    fit = fit_and_summarize(retained_stats, retained_global, args.num_warmup, args.num_samples, args.num_chains, args.seed)
    after = endpoint_ecfp_comparison(fit["stats"], fit["samples"], fit["u_by_endpoint"])

    merged = before.merge(after, on=["endpoint", "embedding"], suffixes=("_before", "_after"))
    merged["delta"] = merged["prob_beat_baseline_after"] - merged["prob_beat_baseline_before"]

    out_dir = os.path.join(OUT_DIR, "test1_truncation")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({k: v for k, v in fit.items() if k not in ("theta_summary", "specialization_summary", "endpoint_summary")},
                os.path.join(out_dir, "posterior.joblib"))
    fit["theta_summary"].to_csv(os.path.join(out_dir, "theta_summary.csv"), index=False)
    merged.to_csv(os.path.join(out_dir, "ecfp_comparison_before_after.csv"), index=False)
    with open(os.path.join(out_dir, "dropped_embeddings.txt"), "w") as f:
        f.write("# theta < cutoff (the truncation test)\n" + "\n".join(drop) + "\n")
        f.write("# excluded for missing raw data on disk (unrelated to the test)\n" + "\n".join(unavailable) + "\n")

    print(f"\nDropped for theta<{THETA_CUTOFF} ({len(drop)}): {drop}")
    if unavailable:
        print(f"Also excluded, missing raw data on disk ({len(unavailable)}): {unavailable}")
    print("\nMeanPredictor theta (null-model ceiling):")
    print(fit["theta_summary"][fit["theta_summary"].embedding == "MeanPredictor"].to_string(index=False))
    print(f"\nP(i>ECFP) movement across {len(merged)} (endpoint, embedding) pairs among retained models:")
    print(f"  mean delta       = {merged['delta'].mean():+.4f}")
    print(f"  mean |delta|     = {merged['delta'].abs().mean():.4f}")
    print(f"  max |delta|      = {merged['delta'].abs().max():.4f}")
    print(f"  sign(mean delta) tells you which way misspecification cuts (positive = retained models look better after dropping laggards)")
    print(f"\nWrote {out_dir}/")


# ---------------------------------------------------------------------------
# Test 2: residual heteroscedasticity
# ---------------------------------------------------------------------------

def cmd_heteroscedastic(args):
    posterior = load_posterior()
    samples = posterior["samples"]
    rows = []
    for stat in posterior["stats"]:
        mu_e = np.asarray(samples[f"mu__{stat.name}"]).mean()
        beta_e = np.asarray(samples[f"beta__{stat.name}"]).mean()
        theta_mean_e = stat.theta_samples.mean(axis=0)
        fitted = mu_e - beta_e * theta_mean_e
        resid = stat.y_hat - fitted
        for j, embedding in enumerate(stat.embeddings):
            rows.append({
                "endpoint": stat.name, "embedding": embedding,
                "y_hat": float(stat.y_hat[j]), "fitted": float(fitted[j]),
                "resid": float(resid[j]), "abs_resid": float(abs(resid[j])),
            })
    df = pd.DataFrame(rows)

    out_dir = os.path.join(OUT_DIR, "test2_heteroscedasticity")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "residuals.csv"), index=False)

    per_endpoint = df.groupby("endpoint").apply(
        lambda g: pd.Series({
            "spearman_rho_yhat_vs_absresid": spearmanr(g["y_hat"], g["abs_resid"]).statistic,
            "n": len(g),
        }),
        include_groups=False,
    ).reset_index()
    per_endpoint.to_csv(os.path.join(out_dir, "spearman_by_endpoint.csv"), index=False)
    overall_rho = spearmanr(df["y_hat"], df["abs_resid"]).statistic

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        endpoints = sorted(df["endpoint"].unique())
        n = len(endpoints)
        ncols = 5
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
        for ax, ep in zip(axes.flat, endpoints):
            g = df[df.endpoint == ep]
            ax.scatter(g["y_hat"], g["abs_resid"], s=10, alpha=0.7)
            ax.set_title(ep.replace("reg_openadmet_", ""), fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.supxlabel("y_hat_ie (measured log MAE)")
        fig.supylabel("|r_ie| (abs residual vs posterior-mean fit)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "heteroscedasticity.png"), dpi=150)
    except ImportError:
        log.warning("matplotlib not available -- skipping plot, CSVs still written")

    print(per_endpoint.sort_values("spearman_rho_yhat_vs_absresid", ascending=False).to_string(index=False))
    print(f"\nPooled Spearman rho(y_hat, |resid|) across all {len(df)} (endpoint, embedding) pairs = {overall_rho:.3f}")
    print(f"Wrote {out_dir}/")


# ---------------------------------------------------------------------------
# Test 3: endpoint correlation / effective sample size
# ---------------------------------------------------------------------------

def cmd_correlation(args):
    posterior = load_posterior()
    global_embeddings = posterior["global_embeddings"]
    stats = posterior["stats"]
    samples = posterior["samples"]

    Y = pd.DataFrame(
        {s.name: pd.Series(s.y_hat, index=s.embeddings).reindex(global_embeddings) for s in stats}
    )
    Z = (Y - Y.mean()) / Y.std(ddof=1)
    empirical = Z.corr()

    endpoints = list(Y.columns)
    a = np.array([np.asarray(samples[f"beta__{e}"]).mean() for e in endpoints])
    omega = np.array([np.asarray(samples[f"omega__{e}"]).mean() for e in endpoints])
    denom = np.sqrt(a**2 + omega**2)
    fitted = pd.DataFrame(np.outer(a, a) / np.outer(denom, denom), index=endpoints, columns=endpoints)

    resid = empirical - fitted
    E = len(endpoints)
    iu = np.triu_indices(E, k=1)
    resid_offdiag = resid.to_numpy()[iu]
    rho_bar = float(resid_offdiag.mean())
    rho_bar_abs = float(np.abs(resid_offdiag).mean())
    # The signed mean can go negative (or below -1/(E-1)) when residual
    # correlation is heterogeneous rather than a uniform positive inflation --
    # E/(1+(E-1)*rho_bar) is only meaningful (in (0, E]) past that point, so
    # also report the abs-mean version as a non-degenerate fallback.
    e_eff = E / (1 + (E - 1) * rho_bar) if (1 + (E - 1) * rho_bar) > 0 else float("nan")
    e_eff_abs = E / (1 + (E - 1) * rho_bar_abs)

    out_dir = os.path.join(OUT_DIR, "test3_correlation")
    os.makedirs(out_dir, exist_ok=True)
    empirical.to_csv(os.path.join(out_dir, "empirical_correlation.csv"))
    fitted.to_csv(os.path.join(out_dir, "fitted_rank1_correlation.csv"))
    resid.to_csv(os.path.join(out_dir, "residual_correlation.csv"))

    print("Residual correlation within flagged blocks (empirical - rank-one fit):")
    for label, members in ENDPOINT_BLOCKS.items():
        members = [m for m in members if m in resid.index]
        sub = resid.loc[members, members]
        off = sub.to_numpy()[np.triu_indices(len(members), k=1)]
        print(f"  {label} ({len(members)} endpoints): mean residual rho = {off.mean():+.3f}, values = {np.round(off, 3).tolist()}")

    print(f"\nResidual off-diagonal correlation: mean(signed) = {rho_bar:+.4f}, mean(|.|) = {rho_bar_abs:.4f}, "
          f"std = {resid_offdiag.std():.4f}, range = [{resid_offdiag.min():+.3f}, {resid_offdiag.max():+.3f}], "
          f"fraction positive = {float((resid_offdiag > 0).mean()):.2f}")
    print(f"E = {E}")
    print(f"  E_eff (signed rho_bar)   = {e_eff:.2f}" + ("  [degenerate: 1+(E-1)*rho_bar <= 0]" if np.isnan(e_eff) else ""))
    print(f"  E_eff (|rho_bar| fallback, conservative) = {e_eff_abs:.2f}")
    print(f"Wrote {out_dir}/")


# ---------------------------------------------------------------------------
# Test 4: log-link refit
# ---------------------------------------------------------------------------

def cmd_loglink(args):
    posterior = load_posterior()
    global_embeddings = posterior["global_embeddings"]

    log_stats = []
    for s in posterior["stats"]:
        D_inv = np.diag(1.0 / s.y_hat)
        V_log = ensure_psd(D_inv @ s.V @ D_inv)
        log_stats.append(EndpointStat(s.name, list(s.embeddings), np.log(s.y_hat), V_log, s.n_compounds))

    log.info(f"Refitting log-link model: {len(global_embeddings)} embeddings across {len(log_stats)} endpoints")
    fit = fit_and_summarize(log_stats, global_embeddings, args.num_warmup, args.num_samples, args.num_chains, args.seed)

    theta_orig = pd.Series(np.asarray(posterior["samples"]["theta"]).mean(axis=0), index=global_embeddings)
    theta_log = fit["theta_summary"].set_index("embedding")["theta_mean"]
    common = theta_orig.index.intersection(theta_log.index)
    tau, p = kendalltau(theta_orig[common], theta_log[common])

    before = endpoint_ecfp_comparison(posterior["stats"], posterior["samples"], posterior["u_by_endpoint"])
    after = endpoint_ecfp_comparison(fit["stats"], fit["samples"], fit["u_by_endpoint"])
    merged = before.merge(after, on=["endpoint", "embedding"], suffixes=("_identity", "_loglink"))

    out_dir = os.path.join(OUT_DIR, "test4_loglink")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({k: v for k, v in fit.items() if k not in ("theta_summary", "specialization_summary", "endpoint_summary")},
                os.path.join(out_dir, "posterior.joblib"))
    fit["theta_summary"].to_csv(os.path.join(out_dir, "theta_summary.csv"), index=False)
    merged.to_csv(os.path.join(out_dir, "table10_identity_vs_loglink.csv"), index=False)

    print(f"Kendall's tau(theta_identity, theta_loglink) = {tau:.4f} (p={p:.2e}) over {len(common)} embeddings")
    print(f"\nP(i>ECFP) movement (identity-link vs log-link), {len(merged)} pairs:")
    diff = merged["prob_beat_baseline_loglink"] - merged["prob_beat_baseline_identity"]
    print(f"  mean = {diff.mean():+.4f}, mean|diff| = {diff.abs().mean():.4f}, max|diff| = {diff.abs().max():.4f}")
    print(f"\nWrote {out_dir}/ (table10_identity_vs_loglink.csv holds both versions of Table 10 side by side)")


# ---------------------------------------------------------------------------
# Addition: omega heteroscedasticity extension log(omega_ie) = log(omega_e) - gamma*theta_i
# ---------------------------------------------------------------------------

def cmd_omega_extension(args):
    posterior = load_posterior()
    global_embeddings = posterior["global_embeddings"]
    stats = posterior["stats"]
    n_embeddings = len(global_embeddings)
    from scipy.linalg import helmert

    import jax.numpy as jnp
    import jax.random as random

    emb_index = {name: i for i, name in enumerate(global_embeddings)}
    Q = jnp.array(helmert(n_embeddings, full=False).T)
    prior_log_s = float(np.log(compute_prior_scale(stats) / np.sqrt(2)))

    def model():
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n_embeddings - 1]).to_event(1))
        theta = numpyro.deterministic("theta", Q @ z)
        sigma_beta = numpyro.sample("sigma_beta", dist.LogNormal(prior_log_s, 0.5))
        sigma_omega = numpyro.sample("sigma_omega", dist.LogNormal(prior_log_s, 0.5))
        gamma = numpyro.sample("gamma", dist.Normal(0.0, 1.0))

        for stat in stats:
            idx = jnp.array([emb_index[e] for e in stat.embeddings])
            mu_e = numpyro.sample(f"mu__{stat.name}", dist.Normal(float(np.mean(stat.y_hat)), 1.0))
            beta_tilde = numpyro.sample(f"beta_tilde__{stat.name}", dist.HalfNormal(1.0))
            omega_tilde = numpyro.sample(f"omega_tilde__{stat.name}", dist.HalfNormal(1.0))
            beta_e = numpyro.deterministic(f"beta__{stat.name}", sigma_beta * beta_tilde)
            omega_e = numpyro.deterministic(f"omega__{stat.name}", sigma_omega * omega_tilde)

            theta_e = theta[idx]
            mean_e = mu_e - beta_e * theta_e
            omega_ie = omega_e * jnp.exp(-gamma * theta_e)
            cov_e = jnp.diag(omega_ie**2) + jnp.array(stat.V)
            numpyro.sample(
                f"y__{stat.name}", dist.MultivariateNormal(mean_e, covariance_matrix=cov_e), obs=jnp.array(stat.y_hat),
            )

    log.info(f"Fitting omega-heteroscedasticity extension: {n_embeddings} embeddings across {len(stats)} endpoints")
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains, progress_bar=False)
    mcmc.run(random.PRNGKey(args.seed))
    mcmc.print_summary(exclude_deterministic=True)
    samples = mcmc.get_samples()

    gamma = np.asarray(samples["gamma"])
    out_dir = os.path.join(OUT_DIR, "test_omega_extension")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "gamma_samples.npy"), gamma)
    print(f"gamma: mean={gamma.mean():.4f}, std={gamma.std():.4f}, P(gamma>0)={float((gamma > 0).mean()):.3f}")
    print("gamma=0 nests the current (shared-omega) model -- a posterior mass away from 0 confirms tests 1/2's signal.")
    print(f"Wrote {out_dir}/")


# ---------------------------------------------------------------------------
# Test 5: endpoint residual correlation Psi_hat
# ---------------------------------------------------------------------------

def zscore_and_kappa(stats, samples):
    """z_ie and kappa_e from app:endpoint-correlation, using posterior means:

        z_ie = (yhat_ie - mu_e + a_e*theta_i) / sqrt(omega_e^2 + vbar_e - cbar_e),
        kappa_e = a_e / omega_e.

    Returns a (union-of-embeddings x endpoints) DataFrame of z (NaN where an
    embedding wasn't fitted on that endpoint) and a Series of kappa_e indexed
    by endpoint name."""
    z_cols, kappa = {}, {}
    for stat in stats:
        mu_e = float(np.asarray(samples[f"mu__{stat.name}"]).mean())
        a_e = float(np.asarray(samples[f"beta__{stat.name}"]).mean())
        omega_e = float(np.asarray(samples[f"omega__{stat.name}"]).mean())
        theta_i = stat.theta_samples.mean(axis=0)

        diag = np.diag(stat.V)
        n = len(stat.embeddings)
        v_bar = diag.mean()
        c_bar = (stat.V.sum() - diag.sum()) / (n * n - n) if n > 1 else 0.0
        resid_scale = np.sqrt(max(omega_e**2 + v_bar - c_bar, EPSILON_0))

        z = (stat.y_hat - mu_e + a_e * theta_i) / resid_scale
        z_cols[stat.name] = pd.Series(z, index=stat.embeddings)
        kappa[stat.name] = a_e / omega_e
    return pd.DataFrame(z_cols), pd.Series(kappa)


def psi_diagnostics(Z, kappa):
    """Psi_hat (pairwise-complete correlation of Z's columns) plus
    lambda_1(Psi_hat), VIF(Psi_hat), E_eff = E/VIF, and the max |off-diagonal|
    (Eq.~eq:endpoint-vif)."""
    Psi_hat = Z.corr()
    Psi_hat = pd.DataFrame(ensure_psd(Psi_hat.to_numpy()), index=Psi_hat.index, columns=Psi_hat.columns)
    endpoints = list(Psi_hat.index)
    kappa_vec = kappa.reindex(endpoints).to_numpy()

    lambda1 = float(np.linalg.eigvalsh(Psi_hat.to_numpy())[-1])
    vif = float(kappa_vec @ Psi_hat.to_numpy() @ kappa_vec / (kappa_vec @ kappa_vec))
    e_eff = len(endpoints) / vif
    iu = np.triu_indices(len(endpoints), k=1)
    max_abs_overall = float(np.abs(Psi_hat.to_numpy()[iu]).max())
    return Psi_hat, lambda1, vif, e_eff, max_abs_overall


def block_max_abs(Psi_hat, blocks):
    out = {}
    for label, members in blocks.items():
        members = [m for m in members if m in Psi_hat.index]
        if len(members) < 2:
            continue
        sub = Psi_hat.loc[members, members].to_numpy()
        out[label] = float(np.abs(sub[np.triu_indices(len(members), k=1)]).max())
    return out


def simulate_null_stats(stats, global_embeddings, samples, rng):
    """One parametric-bootstrap replicate under Psi=I: redraw each endpoint's
    yhat_.e ~ N(mu_e - a_e*theta, omega_e^2*I + V_e) at the fitted posterior
    means, keeping V_e (measurement noise, treated as known) and the
    embedding set per endpoint fixed. Used to calibrate lambda_1(Psi_hat)
    against the null -- fitting mu_e and the rank-one theta term induces some
    residual correlation even when the true Psi is I_E."""
    theta_hat = np.asarray(samples["theta"]).mean(axis=0)
    emb_index = {e: i for i, e in enumerate(global_embeddings)}
    sim_stats = []
    for stat in stats:
        mu_e = float(np.asarray(samples[f"mu__{stat.name}"]).mean())
        a_e = float(np.asarray(samples[f"beta__{stat.name}"]).mean())
        omega_e = float(np.asarray(samples[f"omega__{stat.name}"]).mean())
        idx = [emb_index[e] for e in stat.embeddings]
        mean_e = mu_e - a_e * theta_hat[idx]
        cov_e = omega_e**2 * np.eye(len(idx)) + stat.V
        y_sim = rng.multivariate_normal(mean_e, cov_e)
        sim_stats.append(EndpointStat(stat.name, list(stat.embeddings), y_sim, stat.V, stat.n_compounds))
    return sim_stats


def cmd_endpoint_psi(args):
    posterior = load_posterior()
    stats = posterior["stats"]
    samples = posterior["samples"]
    global_embeddings = posterior["global_embeddings"]

    Z, kappa = zscore_and_kappa(stats, samples)
    Psi_hat, lambda1, vif, e_eff, max_abs_overall = psi_diagnostics(Z, kappa)
    block_max = block_max_abs(Psi_hat, ENDPOINT_BLOCKS)

    out_dir = os.path.join(OUT_DIR, "test5_endpoint_psi")
    os.makedirs(out_dir, exist_ok=True)
    Psi_hat.to_csv(os.path.join(out_dir, "psi_hat.csv"))
    Z.to_csv(os.path.join(out_dir, "z_matrix.csv"))

    print(f"lambda_1(Psi_hat) = {lambda1:.4f}")
    print(f"VIF(Psi_hat) = {vif:.4f}, sqrt(VIF) = {np.sqrt(vif):.4f}")
    print(f"E = {len(Psi_hat)}, E_eff = E/VIF = {e_eff:.2f}")
    print(f"max |Psi_hat_ee'| overall = {max_abs_overall:.4f}")
    for label, val in block_max.items():
        print(f"  max |Psi_hat_ee'| within '{label}' = {val:.4f}")

    log.info(f"Running {args.num_replicates} PPC replicate(s) simulated under Psi=I "
              f"({args.replicate_chains} chain(s) x {args.replicate_warmup}+{args.replicate_samples} draws each)...")
    rng = np.random.default_rng(args.seed)
    lambda1_reps = []
    for r in range(args.num_replicates):
        sim_stats = simulate_null_stats(stats, global_embeddings, samples, rng)
        prior_log_s = float(np.log(compute_prior_scale(sim_stats) / np.sqrt(2)))
        model = build_model(global_embeddings, sim_stats, prior_log_s)
        flat, _ = run_chains(
            model, args.replicate_warmup, args.replicate_samples, args.replicate_chains,
            seed=int(rng.integers(1 << 30)),
        )
        emb_index = {e: i for i, e in enumerate(global_embeddings)}
        theta_rep = np.asarray(flat["theta"])
        for s in sim_stats:
            s.theta_samples = theta_rep[:, [emb_index[e] for e in s.embeddings]]
        Z_rep, kappa_rep = zscore_and_kappa(sim_stats, flat)
        _, lambda1_rep, *_ = psi_diagnostics(Z_rep, kappa_rep)
        lambda1_reps.append(lambda1_rep)
        log.info(f"  replicate {r + 1}/{args.num_replicates}: lambda_1 = {lambda1_rep:.4f}")

    lambda1_reps = np.array(lambda1_reps)
    p_value = float((1 + np.sum(lambda1_reps >= lambda1)) / (1 + len(lambda1_reps)))
    np.save(os.path.join(out_dir, "ppc_lambda1_replicates.npy"), lambda1_reps)

    print(f"\nPPC calibration ({len(lambda1_reps)} replicates, Psi=I, refit through the identical pipeline):")
    print(f"  lambda_1 replicates: mean={lambda1_reps.mean():.4f}, sd={lambda1_reps.std():.4f}, "
          f"range=[{lambda1_reps.min():.4f}, {lambda1_reps.max():.4f}]")
    print(f"  p-value = P(lambda_1_rep >= lambda_1_obs) = {p_value:.4f}")
    print(f"Wrote {out_dir}/")


# ---------------------------------------------------------------------------
# Test 6: equicorrelated Psi sensitivity
# ---------------------------------------------------------------------------

def build_correlated_model(global_embeddings, stats, prior_log_s, rho):
    """build_model extended per app:endpoint-correlation's
    u_{i.} ~ N(0, D_omega*Psi*D_omega), with Psi the equicorrelated matrix
    (1-rho)*I_E + rho*11^T, via the standard factor decomposition

        u_ie = omega_e * (sqrt(1-rho)*eps_ie + sqrt(rho)*xi_i),
        eps_ie ~iid N(0,1) per (i,e), xi_i ~iid N(0,1) per model i (shared
        across every endpoint that model is fit on).

    xi_i is kept as an explicit latent (it carries the cross-endpoint
    coupling for a shared model), but eps_ie is marginalized analytically:
    conditional on xi, it only adds omega_e^2*(1-rho) to endpoint e's
    diagonal covariance, so folding it into cov_e below gives the same
    marginal model at lower latent dimensionality, while keeping that
    covariance's floor at omega_e^2*(1-rho) rather than 0.

    That floor matters operationally: stat.V's eigenvalues run as low as
    ~1e-6 (near-singular pairwise-deletion covariance) while omega_e^2 is
    ~1e-3, so build_model's Sigma_e = omega_e^2*I + V_e relies on the
    omega_e^2 floor to stay well-conditioned. An earlier version sampled eps
    explicitly with covariance_matrix=stat.V alone, dropping the floor to
    V_e's ~1e-6 eigenvalues -- NUTS could not sample the resulting
    near-singular likelihood at all (R-hat up to ~6.8, ESS as low as
    8/64,000 draws at rho>0). rho=0 collapses to build_model's Sigma_e
    exactly."""
    from scipy.linalg import helmert

    import jax.numpy as jnp

    n_embeddings = len(global_embeddings)
    emb_index = {name: i for i, name in enumerate(global_embeddings)}
    Q = jnp.array(helmert(n_embeddings, full=False).T)
    sqrt_rho = float(np.sqrt(rho))

    def model():
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n_embeddings - 1]).to_event(1))
        theta = numpyro.deterministic("theta", Q @ z)
        sigma_beta = numpyro.sample("sigma_beta", dist.LogNormal(prior_log_s, 0.5))
        sigma_omega = numpyro.sample("sigma_omega", dist.LogNormal(prior_log_s, 0.5))
        xi = numpyro.sample("xi", dist.Normal(0.0, 1.0).expand([n_embeddings]).to_event(1))

        for stat in stats:
            idx = jnp.array([emb_index[e] for e in stat.embeddings])
            mu_e = numpyro.sample(f"mu__{stat.name}", dist.Normal(float(np.mean(stat.y_hat)), 1.0))
            beta_tilde = numpyro.sample(f"beta_tilde__{stat.name}", dist.HalfNormal(1.0))
            omega_tilde = numpyro.sample(f"omega_tilde__{stat.name}", dist.HalfNormal(1.0))
            beta_e = numpyro.deterministic(f"beta__{stat.name}", sigma_beta * beta_tilde)
            omega_e = numpyro.deterministic(f"omega__{stat.name}", sigma_omega * omega_tilde)

            theta_e = theta[idx]
            mean_e = mu_e - beta_e * theta_e + omega_e * sqrt_rho * xi[idx]
            cov_e = omega_e**2 * (1.0 - rho) * jnp.eye(len(stat.embeddings)) + jnp.array(stat.V)
            numpyro.sample(
                f"y__{stat.name}",
                dist.MultivariateNormal(mean_e, covariance_matrix=cov_e),
                obs=jnp.array(stat.y_hat),
            )

    return model


def new_endpoint_prob_pair(samples, global_embeddings, emb_a, emb_b, seed):
    """P(emb_a > emb_b) on a hypothetical new endpoint (same construction as
    scripts/visualize_blam.py's compute_ecfp_comparison, specialized to one
    pair) -- also returns the raw per-draw delta_ability samples so callers
    can inspect Var(delta) directly."""
    theta = np.asarray(samples["theta"])
    ia, ib = global_embeddings.index(emb_a), global_embeddings.index(emb_b)
    delta = theta[:, ia] - theta[:, ib]

    rng = np.random.default_rng(seed)
    sigma_beta = np.asarray(samples["sigma_beta"])
    sigma_omega = np.asarray(samples["sigma_omega"])
    beta_new = np.abs(rng.normal(0.0, sigma_beta))
    omega_new = np.abs(rng.normal(0.0, sigma_omega))
    prob = float(ndtr(beta_new * delta / (np.sqrt(2) * omega_new)).mean())
    return prob, delta


def cmd_endpoint_psi_sensitivity(args):
    posterior = load_posterior()
    stats = posterior["stats"]
    global_embeddings = posterior["global_embeddings"]
    samples = posterior["samples"]
    if MONROE not in global_embeddings or BASELINE not in global_embeddings:
        raise SystemExit(f"Need both '{MONROE}' and '{BASELINE}' among fitted embeddings: {global_embeddings}")

    theta_orig = pd.Series(np.asarray(samples["theta"]).mean(axis=0), index=global_embeddings)
    prob_orig, _ = new_endpoint_prob_pair(samples, global_embeddings, MONROE, BASELINE, args.seed)
    print(f"rho=0 [original independent fit]: P({MONROE}>{BASELINE} | new endpoint) = {prob_orig:.4f}")

    # Closed-form equicorrelated bound (Eq.~eq:endpoint-vif specialized to
    # Psi=(1-rho)*I+rho*11^T): VIF(rho) = (1-rho) + rho*wbar, wbar = (sum
    # kappa)^2/sum(kappa^2) in [1, E] by Cauchy-Schwarz -- so VIF(1) = wbar is
    # the rho->1 ceiling, reportable without a refit.
    kappa = np.array([
        float(np.asarray(samples[f"beta__{s.name}"]).mean() / np.asarray(samples[f"omega__{s.name}"]).mean())
        for s in stats
    ])
    w_bar = float(kappa.sum() ** 2 / (kappa**2).sum())
    print(f"\nAnalytic equicorrelated VIF(rho) = (1-rho) + rho*{w_bar:.4f}:")
    for rho in (*RHO_GRID, 1.0):
        vif_rho = (1 - rho) + rho * w_bar
        tag = "  [bound only, no refit]" if rho == 1.0 else ""
        print(f"  rho={rho:.2f}: VIF={vif_rho:.4f}, sqrt(VIF)={np.sqrt(vif_rho):.4f}{tag}")

    out_dir = os.path.join(OUT_DIR, "test6_endpoint_psi_sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    var_at_rho0 = None
    print()
    for rho in sorted(RHO_GRID):
        log.info(f"Refitting equicorrelated model at rho={rho}...")
        prior_log_s = float(np.log(compute_prior_scale(stats) / np.sqrt(2)))
        model = build_correlated_model(global_embeddings, stats, prior_log_s, rho)
        # target_accept_prob=0.9 (vs run_single_chain's 0.8 default): needed for
        # independent chains to agree here -- see run_single_chain's docstring.
        flat, _ = run_chains(
            model, args.num_warmup, args.num_samples, args.num_chains, args.seed,
            target_accept_prob=0.9, filter_outliers=True,
        )

        prob, delta = new_endpoint_prob_pair(flat, global_embeddings, MONROE, BASELINE, args.seed)
        var_delta = float(delta.var(ddof=1))
        if rho == 0.0:
            var_at_rho0 = var_delta

        theta_rho = pd.Series(np.asarray(flat["theta"]).mean(axis=0), index=global_embeddings)
        rho_spearman, _ = spearmanr(theta_orig, theta_rho)

        rows.append({
            "rho": rho,
            "prob_monroe_beat_ecfp_new_endpoint": prob,
            "var_delta_monroe_ecfp": var_delta,
            "empirical_vif_vs_rho0": var_delta / var_at_rho0,
            "spearman_vs_table2": float(rho_spearman),
        })
        print(f"rho={rho:.2f}: P({MONROE}>{BASELINE}|new endpoint)={prob:.4f}, "
              f"empirical VIF vs rho=0={var_delta / var_at_rho0:.4f}, "
              f"Spearman(theta, Table 2)={rho_spearman:.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "sensitivity_by_rho.csv"), index=False)
    print("\n(rho=0 collapses in distribution to the original independent model, so its P and Spearman "
          "above are a sanity check against the '[original independent fit]' row.)")
    print(f"Wrote {out_dir}/")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    subparsers = {}
    for name, fn in [
        ("truncation", cmd_truncation), ("heteroscedastic", cmd_heteroscedastic),
        ("correlation", cmd_correlation), ("loglink", cmd_loglink), ("omega_extension", cmd_omega_extension),
        ("endpoint_psi", cmd_endpoint_psi), ("endpoint_psi_sensitivity", cmd_endpoint_psi_sensitivity),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--num-warmup", type=int, default=1000)
        p.add_argument("--num-samples", type=int, default=2000)
        p.add_argument("--num-chains", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(func=fn)
        subparsers[name] = p

    subparsers["endpoint_psi"].add_argument(
        "--num-replicates", type=int, default=100,
        help="PPC replicates simulated under Psi=I and refit through the identical pipeline",
    )
    subparsers["endpoint_psi"].add_argument("--replicate-warmup", type=int, default=500)
    subparsers["endpoint_psi"].add_argument("--replicate-samples", type=int, default=500)
    subparsers["endpoint_psi"].add_argument("--replicate-chains", type=int, default=1)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    numpyro.set_host_device_count(args.num_chains)
    args.func(args)
