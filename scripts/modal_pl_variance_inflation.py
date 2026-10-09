"""Run scripts/pl_variance_inflation.py's scenario F (does the actual BBT/NUTS
posterior, not just the analytic Fisher-info variance, undercover?) at the
same reps=3000 scale as scenarios A-E.

Each replicate is a full independent NUTS fit (~1.3s locally after JIT
warmup), so 3000 reps single-threaded would take well over an hour. Unlike
scripts/modal_blam_analysis.py's chain fan-out -- where every chain samples
the *same* posterior and has to be recombined for R-hat/n_eff -- each
replicate here is its own independent Monte Carlo draw (fresh Plackett-Luce
data, fresh posterior), so fan-out is a plain embarrassingly-parallel split:
num_workers containers each run reps/num_workers replicates, and the results
are just concatenated before computing coverage/VIF.

Usage:
    modal run --detach scripts/modal_pl_variance_inflation.py
    modal run --detach scripts/modal_pl_variance_inflation.py --reps 3000 --num-workers 50
    modal run --detach scripts/modal_pl_variance_inflation.py --non-null
    modal run --detach scripts/modal_pl_variance_inflation.py --m 20 --d 100

Run detached for the same reason as modal_score_errors.py/modal_blam_analysis.py:
a non-detached `modal run` tears down every in-flight container if the local
client disconnects mid-run.
"""

import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("ml-embedding-benchmark-pl-variance-inflation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy",
        # Same jax/jaxlib/numpyro pins as requirements.txt.
        "jax==0.4.31",
        "jaxlib==0.4.31",
        "numpyro==0.15.3",
    )
    # env() before add_local_file: Modal requires build steps like this to
    # precede add_local_* or the setting only takes effect at container
    # startup, too late for the sys.path-dependent import below.
    .env({"PYTHONPATH": "/project/scripts"})
    .add_local_file(
        PROJECT_ROOT / "scripts" / "pl_variance_inflation.py",
        remote_path="/project/scripts/pl_variance_inflation.py",
    )
)


def _run_batch(M: int, D: int, reps: int, worker_idx: int, non_null: bool, num_warmup: int, num_samples: int):
    import numpy as np
    from pl_variance_inflation import simulate_bayes_batch

    beta_true = np.linspace(-1.5, 1.5, M) if non_null else None
    rng = np.random.default_rng(20260826 + worker_idx)
    # seed0 offset by worker_idx * reps (the *target* total, not this
    # worker's own share) so no two workers' PRNGKeys collide even when reps
    # doesn't divide evenly across num_workers.
    return simulate_bayes_batch(
        M, D, reps, rng, beta_true=beta_true,
        num_warmup=num_warmup, num_samples=num_samples, seed0=worker_idx * reps,
    )


def _finalize(batches: list, M: int, D: int, beta_true_desc: str) -> dict:
    import numpy as np
    from pl_variance_inflation import report_bayes, summarize_bayes

    hats = np.concatenate([np.asarray(b[0]) for b in batches], axis=0)
    v_post = np.concatenate([np.asarray(b[1]) for b in batches], axis=0)
    tr_post = np.concatenate([np.asarray(b[2]) for b in batches], axis=0)
    beta_true = np.linspace(-1.5, 1.5, M) if beta_true_desc == "non-null" else np.zeros(M)

    res = summarize_bayes(hats, v_post, tr_post, beta_true)
    report_bayes(f"F. Plackett-Luce, NUTS posterior ({beta_true_desc} abilities)", res, M, D, len(hats))
    return res


# Each worker runs its own share of replicates sequentially, one full NUTS
# fit at a time -- no benefit to more than a couple of cores per container.
WORKER_CPU = 2
WORKER_TIMEOUT = 60 * 60 * 2

run_batch_fn = app.function(image=image, timeout=WORKER_TIMEOUT, cpu=WORKER_CPU)(_run_batch)
finalize_fn = app.function(image=image, timeout=600, cpu=2)(_finalize)


@app.local_entrypoint()
def main(
    reps: int = 3000,
    num_workers: int = 50,
    m: int = 10,
    d: int = 200,
    non_null: bool = False,
    num_warmup: int = 300,
    num_samples: int = 500,
):
    beta_true_desc = "non-null" if non_null else "null"
    counts = [reps // num_workers + (1 if i < reps % num_workers else 0) for i in range(num_workers)]
    counts = [c for c in counts if c > 0]

    print(f"Fanning out {reps} replicate(s) ({beta_true_desc} abilities) across {len(counts)} container(s)...")
    calls = [
        run_batch_fn.spawn(m, d, c, worker_idx, non_null, num_warmup, num_samples)
        for worker_idx, c in enumerate(counts)
    ]
    batches = [call.get() for call in calls]

    print("Combining batches and summarizing...")
    finalize_fn.remote(batches, m, d, beta_true_desc)
