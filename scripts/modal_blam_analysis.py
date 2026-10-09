"""Run scripts/blam_analysis.py's NUTS fit on Modal, against data/errors
already on the Volume (put there by scripts/modal_score_errors.py).

HMC chains are embarrassingly parallel -- no cross-chain communication during
sampling, only combined afterward for point estimates and R-hat/n_eff
diagnostics -- so each chain gets its own container with its own dedicated
CPU, instead of numpyro's own num_chains-in-one-process parallelism, where
every chain fights the same container's cores for the whole run (observed
locally: with num_chains=4 on an 8-core box, one chain reached 80% while
another sat at 5%, entirely from cross-chain contention, not the model
itself). Splitting across containers removes that contention rather than
just moving it.

Fan-out is 3 stages: one `prepare` call loads data/errors once and builds the
model (cheap, deterministic -- redoing it per chain container would be
redundant Volume I/O for no benefit); num_chains independent `run_chain`
calls, each a full NUTS chain from its own PRNGKey; one `finalize` call
recombines every chain's post-warmup draws (numpyro.diagnostics.summary()
needs chains kept separate -- grouped as (num_chains, num_samples, ...) --
to compute r_hat/n_eff; the same draws flattened across chains is what
compute_specialization()/summarize() need) and writes results to the Volume.

Usage:
    modal run --detach scripts/modal_blam_analysis.py
    modal run --detach scripts/modal_blam_analysis.py --num-chains 8 --num-samples 4000
    modal run --detach scripts/modal_blam_analysis.py --dataset-glob reg_openadmet_pxr_pEC50

Run detached for the same reason as modal_score_errors.py: a non-detached
`modal run` tears down every in-flight container if the local client
disconnects mid-fit. After it completes, pull results back:

    modal volume get ml-embedding-benchmark /blam data/blam
"""

import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VOLUME_NAME = "ml-embedding-benchmark"
MOUNT = pathlib.Path("/data")

app = modal.App("ml-embedding-benchmark-blam")
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
    # Set at the image level, not inside each function body: `stats` (a list
    # of EndpointStat, defined in blam_analysis.py) gets deserialized by
    # Modal's container entrypoint before any function code runs, so
    # sys.path.insert() inside the function body would be too late. Must
    # precede add_local_file -- Modal requires build steps like .env() to
    # come before add_local_*, or the setting only applies at container
    # startup.
    .env({"PYTHONPATH": "/project/scripts"})
    .add_local_file(
        PROJECT_ROOT / "scripts" / "blam_analysis.py",
        remote_path="/project/scripts/blam_analysis.py",
    )
)


def _prepare(dataset_glob: str, methods: str, embeddings: str = None, finetuned_models: str = None):
    import numpy as np
    from blam_analysis import compute_prior_scale, load_endpoint_stats

    volume.reload()
    methods_t = tuple(m.strip() for m in methods.split(",") if m.strip())
    allowed = frozenset(e.strip() for e in embeddings.split(",") if e.strip()) if embeddings else None
    finetuned_specs = None
    if finetuned_models:
        finetuned_specs = []
        for entry in (e.strip() for e in finetuned_models.split(",") if e.strip()):
            entity, file_stem = entry.split("=", 1) if "=" in entry else (entry, entry)
            finetuned_specs.append((f"{entity}_finetuned", file_stem))
    stats = load_endpoint_stats(
        str(MOUNT / "errors"), dataset_glob, methods_t, allowed, str(MOUNT / "finetune_errors"), finetuned_specs
    )
    if not stats:
        raise ValueError("No endpoints had >=2 embeddings with full method coverage -- nothing to fit")
    global_embeddings = sorted({e for stat in stats for e in stat.embeddings})
    prior_log_s = float(np.log(compute_prior_scale(stats) / np.sqrt(2)))
    return stats, global_embeddings, prior_log_s


def _run_chain(stats: list, global_embeddings: list, prior_log_s: float, num_warmup: int, num_samples: int, seed: int) -> dict:
    from blam_analysis import build_model, run_single_chain

    model = build_model(global_embeddings, stats, prior_log_s)
    return run_single_chain(model, seed, num_warmup, num_samples)


def _finalize(stats: list, global_embeddings: list, chain_samples: list, output_subdir: str, specialization_seed: int) -> str:
    import joblib
    import numpy as np
    from blam_analysis import combine_chains, compute_specialization, summarize
    from numpyro.diagnostics import print_summary

    flat_samples, grouped_samples = combine_chains(chain_samples)
    print_summary(grouped_samples, prob=0.9)

    emb_index = {name: i for i, name in enumerate(global_embeddings)}
    theta = np.asarray(flat_samples["theta"])
    for stat in stats:
        idx = [emb_index[e] for e in stat.embeddings]
        stat.theta_samples = theta[:, idx]

    u_by_endpoint = compute_specialization(flat_samples, stats, specialization_seed)
    theta_summary, specialization_summary, endpoint_summary = summarize(
        global_embeddings, flat_samples, stats, u_by_endpoint
    )

    volume.reload()
    output_dir = MOUNT / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"samples": flat_samples, "grouped_samples": grouped_samples, "stats": stats,
         "u_by_endpoint": u_by_endpoint, "global_embeddings": global_embeddings},
        str(output_dir / "posterior.joblib"),
    )
    theta_summary.to_csv(str(output_dir / "theta_summary.csv"), index=False)
    specialization_summary.to_csv(str(output_dir / "specialization_summary.csv"), index=False)
    endpoint_summary.to_csv(str(output_dir / "endpoint_summary.csv"), index=False)
    volume.commit()

    return "Latent ability ranking (theta, higher = better):\n" + theta_summary.to_string(index=False)


PREPARE_CPU = 2
# Each chain runs alone in its own container -- no sibling chains competing
# for the same cores, unlike num_chains-in-one-process. 2 cores gives JAX's
# own internal linalg ops room without paying for cores that would sit idle
# (a single NUTS chain doesn't parallelize across many cores on its own).
CHAIN_CPU = 2
FINALIZE_CPU = 2
CHAIN_TIMEOUT = 60 * 60 * 2
# _prepare reads every (embedding, method) joblib per endpoint individually off
# the Volume -- ~45-50s/endpoint observed (network I/O per file, not compute)
# vs ~0.3s/endpoint reading the same files off local disk -- so 17 endpoints
# needs well over the default 600s.
PREPARE_TIMEOUT = 1800

prepare_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=PREPARE_TIMEOUT, cpu=PREPARE_CPU)(_prepare)
run_chain_fn = app.function(image=image, timeout=CHAIN_TIMEOUT, cpu=CHAIN_CPU)(_run_chain)
finalize_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=600, cpu=FINALIZE_CPU)(_finalize)


@app.local_entrypoint()
def main(
    dataset_glob: str = "reg_openadmet_*",
    methods: str = "rf,ridge,knn,tabicl,tabpfn",
    embeddings: str = None,
    embeddings_file: str = None,
    finetuned_models: str = None,
    finetuned_models_file: str = None,
    num_warmup: int = 2000,
    num_samples: int = 4000,
    num_chains: int = 16,
    seed: int = 0,
    specialization_seed: int = 0,
    output_subdir: str = "blam",
):
    if embeddings_file:
        embeddings = ",".join(line.strip() for line in pathlib.Path(embeddings_file).read_text().splitlines() if line.strip())
    if finetuned_models_file:
        finetuned_models = ",".join(line.strip() for line in pathlib.Path(finetuned_models_file).read_text().splitlines() if line.strip())

    print("Loading data/errors and building the model (one container)...")
    stats, global_embeddings, prior_log_s = prepare_fn.remote(dataset_glob, methods, embeddings, finetuned_models)
    print(f"Fitting {len(global_embeddings)} embeddings across {len(stats)} endpoints")

    print(f"Fanning out {num_chains} independent chain(s), one container each...")
    chain_calls = [
        run_chain_fn.spawn(stats, global_embeddings, prior_log_s, num_warmup, num_samples, seed + i)
        for i in range(num_chains)
    ]
    chain_samples = [call.get() for call in chain_calls]

    print("Combining chains and finalizing (one container)...")
    print(finalize_fn.remote(stats, global_embeddings, chain_samples, output_subdir, specialization_seed))
