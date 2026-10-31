"""Run scripts/blam_robustness_checks.py's endpoint_psi and
endpoint_psi_sensitivity tests (endpoint residual correlation) on
Modal, against an already-fitted posterior.joblib already on the Volume (put
there by scripts/modal_blam_analysis.py's `finalize` stage, blam/posterior.joblib
by default).

Two independent local entrypoints, one per test, both following the same
3-stage fan-out as modal_blam_analysis.py: one `prepare` call loads the
posterior once and computes anything cheap (no refit needed); many
independent `run_*` calls, each its own container running one NUTS chain (or,
for endpoint_psi's PPC, one full replicate: simulate + refit); one `finalize`
call combines results and writes them back to the Volume.

  endpoint_psi:             fans out over PPC replicates (each an independent
                             container: simulate under Psi=I, refit, compute
                             lambda_1(Psi_hat_rep)).
  endpoint_psi_sensitivity: fans out over (rho, chain) pairs -- num_chains
                             independent containers per rho in {0,.25,.5,.75},
                             all rho values run concurrently.

Usage:
    modal run --detach scripts/modal_blam_robustness_checks.py::app.endpoint_psi
    modal run --detach scripts/modal_blam_robustness_checks.py::app.endpoint_psi --num-replicates 200
    modal run --detach scripts/modal_blam_robustness_checks.py::app.endpoint_psi_sensitivity
    modal run --detach scripts/modal_blam_robustness_checks.py::app.endpoint_psi_sensitivity --num-chains 16 --posterior-subdir blam

Run detached for the same reason as modal_blam_analysis.py: a non-detached
`modal run` tears down every in-flight container if the local client
disconnects mid-fit. After it completes, pull results back:

    modal volume get ml-embedding-benchmark /blam_tests data/blam_tests
"""

import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VOLUME_NAME = "ml-embedding-benchmark"
MOUNT = pathlib.Path("/data")
POSTERIOR_SUBDIR = "blam"
# Must match blam_robustness_checks.py's constants of the same name --
# duplicated here only so the local entrypoints can print/loop over them
# without importing a jax/numpyro-dependent module into the local process
# just for three constants; the actual computation always imports the real
# ones from blam_robustness_checks inside the remote function bodies below.
RHO_GRID = (0.0, 0.25, 0.5, 0.75)
MONROE = "monroe"
BASELINE = "ECFP_count"

app = modal.App("ml-embedding-benchmark-blam-robustness")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "pandas==2.3.3",
        "scipy",
        "joblib==1.4.2",
        # Same jax/jaxlib/numpyro pins as requirements.txt.
        "jax==0.4.31",
        "jaxlib==0.4.31",
        "numpyro==0.15.3",
    )
    # See modal_blam_analysis.py's image comment: must precede add_local_file,
    # and blam_robustness_checks.py itself does `from blam_analysis import
    # ...` at module scope, so both files need to land here before either is
    # imported by a container's entrypoint machinery.
    .env({"PYTHONPATH": "/project/scripts"})
    .add_local_file(PROJECT_ROOT / "scripts" / "blam_analysis.py", remote_path="/project/scripts/blam_analysis.py")
    .add_local_file(
        PROJECT_ROOT / "scripts" / "blam_robustness_checks.py",
        remote_path="/project/scripts/blam_robustness_checks.py",
    )
)


def _strip_theta_samples(stats):
    """Fresh EndpointStat copies without the theta_samples attribute the
    stored fit's stats carry (blam_analysis.main() attaches it before
    dumping posterior.joblib) -- every refit container below computes its
    own theta_samples from its own chain, so shipping the original fit's
    copy to each one would just be wasted transfer (48 x n_e x 8 bytes each,
    repeated once per replicate/chain container)."""
    from blam_analysis import EndpointStat

    return [EndpointStat(s.name, list(s.embeddings), s.y_hat, s.V, s.n_compounds) for s in stats]


def _load_posterior(posterior_subdir: str):
    import joblib

    volume.reload()
    return joblib.load(str(MOUNT / posterior_subdir / "posterior.joblib"))


# ---------------------------------------------------------------------------
# endpoint_psi
# ---------------------------------------------------------------------------

def _prepare_psi(posterior_subdir: str):
    import numpy as np
    from blam_robustness_checks import ENDPOINT_BLOCKS, block_max_abs, psi_diagnostics, zscore_and_kappa

    posterior = _load_posterior(posterior_subdir)
    stats, samples, global_embeddings = posterior["stats"], posterior["samples"], posterior["global_embeddings"]

    Z, kappa = zscore_and_kappa(stats, samples)
    Psi_hat, lambda1, vif, e_eff, max_abs_overall = psi_diagnostics(Z, kappa)
    block_max = block_max_abs(Psi_hat, ENDPOINT_BLOCKS)

    point_estimates = {
        "theta_hat": np.asarray(samples["theta"]).mean(axis=0),
        "endpoint_params": {
            s.name: {
                "mu": float(np.asarray(samples[f"mu__{s.name}"]).mean()),
                "a": float(np.asarray(samples[f"a__{s.name}"]).mean()),
                "omega": float(np.asarray(samples[f"omega__{s.name}"]).mean()),
            }
            for s in stats
        },
    }
    diagnostics = {
        "lambda1": lambda1, "vif": vif, "e_eff": e_eff, "max_abs_overall": max_abs_overall,
        "block_max": block_max, "Z": Z, "Psi_hat": Psi_hat,
    }
    return _strip_theta_samples(stats), global_embeddings, point_estimates, diagnostics


def _run_ppc_replicate(
    stats: list, global_embeddings: list, point_estimates: dict,
    num_warmup: int, num_samples: int, num_chains: int, seed: int,
) -> float:
    """One PPC replicate: simulate yhat_.e under Psi=I at the fitted
    posterior means (point_estimates), refit through the identical
    (independent) pipeline, and return lambda_1(Psi_hat_rep)."""
    import numpy as np
    from blam_analysis import EndpointStat, build_model, compute_prior_scale
    from blam_robustness_checks import psi_diagnostics, run_chains, zscore_and_kappa

    rng = np.random.default_rng(seed)
    emb_index = {e: i for i, e in enumerate(global_embeddings)}
    theta_hat = point_estimates["theta_hat"]

    sim_stats = []
    for stat in stats:
        p = point_estimates["endpoint_params"][stat.name]
        idx = [emb_index[e] for e in stat.embeddings]
        mean_e = p["mu"] - p["a"] * theta_hat[idx]
        cov_e = p["omega"] ** 2 * np.eye(len(idx)) + stat.V
        y_sim = rng.multivariate_normal(mean_e, cov_e)
        sim_stats.append(EndpointStat(stat.name, list(stat.embeddings), y_sim, stat.V, stat.n_compounds))

    prior_log_s = float(np.log(compute_prior_scale(sim_stats) / np.sqrt(2)))
    model = build_model(global_embeddings, sim_stats, prior_log_s)
    flat, _ = run_chains(model, num_warmup, num_samples, num_chains, seed)

    theta_rep = np.asarray(flat["theta"])
    for s in sim_stats:
        s.theta_samples = theta_rep[:, [emb_index[e] for e in s.embeddings]]
    _, lambda1_rep, *_ = psi_diagnostics(*zscore_and_kappa(sim_stats, flat))
    return float(lambda1_rep)


def _finalize_psi(diagnostics: dict, lambda1_reps: list, output_subdir: str) -> str:
    import numpy as np

    lambda1_reps = np.array(lambda1_reps)
    p_value = float((1 + np.sum(lambda1_reps >= diagnostics["lambda1"])) / (1 + len(lambda1_reps)))

    volume.reload()
    output_dir = MOUNT / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics["Psi_hat"].to_csv(str(output_dir / "psi_hat.csv"))
    diagnostics["Z"].to_csv(str(output_dir / "z_matrix.csv"))
    np.save(str(output_dir / "ppc_lambda1_replicates.npy"), lambda1_reps)
    volume.commit()

    lines = [
        f"lambda_1(Psi_hat) = {diagnostics['lambda1']:.4f}",
        f"VIF(Psi_hat) = {diagnostics['vif']:.4f}, sqrt(VIF) = {np.sqrt(diagnostics['vif']):.4f}",
        f"E = {len(diagnostics['Psi_hat'])}, E_eff = E/VIF = {diagnostics['e_eff']:.2f}",
        f"max |Psi_hat_ee'| overall = {diagnostics['max_abs_overall']:.4f}",
    ]
    for label, val in diagnostics["block_max"].items():
        lines.append(f"  max |Psi_hat_ee'| within '{label}' = {val:.4f}")
    lines.append(
        f"\nPPC calibration ({len(lambda1_reps)} replicates, Psi=I, refit through the identical pipeline): "
        f"lambda_1 mean={lambda1_reps.mean():.4f}, sd={lambda1_reps.std():.4f}, "
        f"range=[{lambda1_reps.min():.4f}, {lambda1_reps.max():.4f}]"
    )
    lines.append(f"p-value = P(lambda_1_rep >= lambda_1_obs) = {p_value:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# endpoint_psi_sensitivity
# ---------------------------------------------------------------------------

def _prepare_sensitivity(posterior_subdir: str):
    import numpy as np
    from blam_analysis import compute_prior_scale
    from blam_robustness_checks import BASELINE, MONROE, new_endpoint_prob_pair

    posterior = _load_posterior(posterior_subdir)
    stats, samples, global_embeddings = posterior["stats"], posterior["samples"], posterior["global_embeddings"]

    theta_orig = np.asarray(samples["theta"]).mean(axis=0)
    kappa = np.array([
        float(np.asarray(samples[f"a__{s.name}"]).mean() / np.asarray(samples[f"omega__{s.name}"]).mean())
        for s in stats
    ])
    prior_log_s = float(np.log(compute_prior_scale(stats) / np.sqrt(2)))
    prob_orig, _ = new_endpoint_prob_pair(samples, global_embeddings, MONROE, BASELINE, seed=0)

    return _strip_theta_samples(stats), global_embeddings, prior_log_s, theta_orig, kappa, prob_orig


def _run_chain_rho(stats: list, global_embeddings: list, prior_log_s: float, rho: float,
                    num_warmup: int, num_samples: int, seed: int) -> dict:
    from blam_analysis import run_single_chain
    from blam_robustness_checks import build_correlated_model

    model = build_correlated_model(global_embeddings, stats, prior_log_s, rho)
    # target_accept_prob=0.9: see run_single_chain's docstring -- independent
    # chains disagree far outside their own posterior sd at the 0.8 default
    # (xi_i sits in a near-degenerate ridge with theta_i).
    return run_single_chain(model, seed, num_warmup, num_samples, target_accept_prob=0.9)


def _finalize_rho(global_embeddings: list, chain_samples: list, rho: float, theta_orig, output_subdir: str) -> dict:
    """Combine one rho's chains, compute P(monroe>ECFP) on a new endpoint and
    the Spearman rank correlation of theta against the rho=0 (Table 2)
    ranking, and persist that rho's posterior to the Volume."""
    import logging as log

    import joblib
    import numpy as np
    from scipy.stats import spearmanr

    from blam_analysis import combine_chains
    from blam_robustness_checks import BASELINE, MONROE, filter_outlier_chains, new_endpoint_prob_pair

    if len(chain_samples) >= 4:
        chain_samples, dropped = filter_outlier_chains(chain_samples)
        if dropped:
            log.warning(
                f"rho={rho}: dropped {len(dropped)} outlier chain(s) (indices {dropped}) -- "
                "see filter_outlier_chains docstring"
            )
    flat, grouped = combine_chains(chain_samples)
    prob, delta = new_endpoint_prob_pair(flat, global_embeddings, MONROE, BASELINE, seed=0)
    var_delta = float(delta.var(ddof=1))
    theta_rho = np.asarray(flat["theta"]).mean(axis=0)
    rho_spearman, _ = spearmanr(theta_orig, theta_rho)

    volume.reload()
    output_dir = MOUNT / output_subdir / f"rho_{rho:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"samples": flat, "grouped_samples": grouped, "global_embeddings": global_embeddings,
         "n_chains_kept": len(chain_samples)},
        str(output_dir / "posterior.joblib"),
    )
    volume.commit()

    return {
        "rho": rho, "prob_monroe_beat_ecfp_new_endpoint": prob,
        "var_delta_monroe_ecfp": var_delta, "spearman_vs_table2": float(rho_spearman),
    }


def _write_sensitivity_summary(rows: list, w_bar: float, prob_orig: float, output_subdir: str) -> str:
    import numpy as np
    import pandas as pd

    rows = sorted(rows, key=lambda r: r["rho"])
    var_at_rho0 = next(r["var_delta_monroe_ecfp"] for r in rows if r["rho"] == 0.0)
    for r in rows:
        r["empirical_vif_vs_rho0"] = r["var_delta_monroe_ecfp"] / var_at_rho0
    df = pd.DataFrame(rows)[[
        "rho", "prob_monroe_beat_ecfp_new_endpoint", "var_delta_monroe_ecfp",
        "empirical_vif_vs_rho0", "spearman_vs_table2",
    ]]

    volume.reload()
    output_dir = MOUNT / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(output_dir / "sensitivity_by_rho.csv"), index=False)
    volume.commit()

    lines = [
        f"rho=0 [original independent fit]: P({MONROE}>{BASELINE} | new endpoint) = {prob_orig:.4f}",
        "",
        f"Analytic equicorrelated VIF(rho) = (1-rho) + rho*{w_bar:.4f}:",
    ]
    for rho in (*RHO_GRID, 1.0):
        vif_rho = (1 - rho) + rho * w_bar
        tag = "  [bound only, no refit]" if rho == 1.0 else ""
        lines.append(f"  rho={rho:.2f}: VIF={vif_rho:.4f}, sqrt(VIF)={np.sqrt(vif_rho):.4f}{tag}")
    lines.append("")
    for _, r in df.iterrows():
        lines.append(
            f"rho={r.rho:.2f}: P({MONROE}>{BASELINE}|new endpoint)={r.prob_monroe_beat_ecfp_new_endpoint:.4f}, "
            f"empirical VIF vs rho=0={r.empirical_vif_vs_rho0:.4f}, Spearman(theta, Table 2)={r.spearman_vs_table2:.4f}"
        )
    return "\n".join(lines)


PREPARE_CPU = 2
# Each chain/replicate runs alone in its own container -- see
# modal_blam_analysis.py's CHAIN_CPU comment for why (no sibling chains
# competing for the same cores the way numpyro's own num_chains-in-one-process
# parallelism does).
CHAIN_CPU = 2
FINALIZE_CPU = 2
CHAIN_TIMEOUT = 60 * 60 * 2
PREPARE_TIMEOUT = 600

prepare_psi_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=PREPARE_TIMEOUT, cpu=PREPARE_CPU)(_prepare_psi)
run_ppc_replicate_fn = app.function(image=image, timeout=CHAIN_TIMEOUT, cpu=CHAIN_CPU)(_run_ppc_replicate)
finalize_psi_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=600, cpu=FINALIZE_CPU)(_finalize_psi)

prepare_sensitivity_fn = app.function(
    image=image, volumes={str(MOUNT): volume}, timeout=PREPARE_TIMEOUT, cpu=PREPARE_CPU,
)(_prepare_sensitivity)
run_chain_rho_fn = app.function(image=image, timeout=CHAIN_TIMEOUT, cpu=CHAIN_CPU)(_run_chain_rho)
finalize_rho_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=600, cpu=FINALIZE_CPU)(_finalize_rho)
write_sensitivity_summary_fn = app.function(
    image=image, volumes={str(MOUNT): volume}, timeout=300, cpu=1,
)(_write_sensitivity_summary)


@app.local_entrypoint(name="endpoint_psi")
def main_endpoint_psi(
    posterior_subdir: str = POSTERIOR_SUBDIR,
    num_replicates: int = 100,
    replicate_warmup: int = 500,
    replicate_samples: int = 500,
    replicate_chains: int = 1,
    seed: int = 0,
    output_subdir: str = "blam_tests/test5_endpoint_psi",
):
    print(f"Loading posterior from '{posterior_subdir}' and computing observed Psi_hat (one container)...")
    stats, global_embeddings, point_estimates, diagnostics = prepare_psi_fn.remote(posterior_subdir)
    print(f"lambda_1(Psi_hat) = {diagnostics['lambda1']:.4f}, VIF = {diagnostics['vif']:.4f}")

    print(f"Fanning out {num_replicates} PPC replicate(s), one container each...")
    calls = [
        run_ppc_replicate_fn.spawn(
            stats, global_embeddings, point_estimates, replicate_warmup, replicate_samples, replicate_chains, seed + i,
        )
        for i in range(num_replicates)
    ]
    lambda1_reps = [call.get() for call in calls]

    print("Finalizing (one container)...")
    print(finalize_psi_fn.remote(diagnostics, lambda1_reps, output_subdir))


@app.local_entrypoint(name="endpoint_psi_sensitivity")
def main_endpoint_psi_sensitivity(
    posterior_subdir: str = POSTERIOR_SUBDIR,
    num_warmup: int = 2000,
    num_samples: int = 4000,
    num_chains: int = 16,
    seed: int = 0,
    output_subdir: str = "blam_tests/test6_endpoint_psi_sensitivity",
):
    print(f"Loading posterior from '{posterior_subdir}' (one container)...")
    stats, global_embeddings, prior_log_s, theta_orig, kappa, prob_orig = prepare_sensitivity_fn.remote(posterior_subdir)
    w_bar = float(kappa.sum() ** 2 / (kappa**2).sum())

    print(f"Fanning out {num_chains} chain(s) x {len(RHO_GRID)} rho value(s) "
          f"= {num_chains * len(RHO_GRID)} container(s), all rho concurrently...")
    chain_calls = {
        rho: [
            run_chain_rho_fn.spawn(stats, global_embeddings, prior_log_s, rho, num_warmup, num_samples, seed + i)
            for i in range(num_chains)
        ]
        for rho in RHO_GRID
    }

    print("Finalizing each rho (one container each)...")
    rows = [
        finalize_rho_fn.remote(global_embeddings, [c.get() for c in calls], rho, theta_orig, output_subdir)
        for rho, calls in chain_calls.items()
    ]

    print("Writing combined summary...")
    print(write_sensitivity_summary_fn.remote(rows, w_bar, prob_orig, output_subdir))
