"""Fan out scripts/compute_prediction_errors.py across Modal containers -- one
container per (embedding, method, endpoint) triple. Unlike scripts/modal_embed.py
(one image per wrapper, each with its own pinned legacy deps), regression
scoring only ever needs sklearn + scikit-fingerprints, so every triple shares a
single image -- except 'tabicl', 'tabpfn', and 'tabpfn_v2' (see below), which
need their own GPU images.

'tabicl' (https://github.com/soda-inria/tabicl) and 'tabpfn'/'tabpfn_v2'
(https://github.com/PriorLabs/TabPFN) are pretrained tabular in-context-
learning models rather than fitted sklearn estimators, so each runs on its own
GPU-attached image/function built with torch + the respective package.
Everything else about the fan-out (grain, resumability, Volume layout) is
identical -- compute_prediction_errors.py dispatches on --method the same way
regardless.

'tabpfn' and 'tabpfn_v2' both come from the same `tabpfn` package
(src/eval/supervised/tabpfn_head.py) but pin different pretrained model
generations: 'tabpfn' is TabPFNRegressor's own "auto" default, which resolves
to the v3 model -- our official, most-up-to-date results. 'tabpfn_v2' pins
the older v2 model explicitly (TabPFNRegressor.create_default_for_version),
whose checkpoint is Apache-licensed with no commercial-use restriction, unlike
v3's Prior Labs license -- it's run separately for internal-only benchmarking
comparisons, not to replace the v3 numbers.

tabpfn's (v3) default checkpoint repo is gated on HuggingFace (requires an
authenticated, approved HF account), so its image build authenticates via the
'hf-ne' Modal secret (HF_TOKEN) to prefetch it -- see
_prefetch_tabpfn_checkpoint for why downloading it this way (rather than
letting tabpfn's own TabPFNRegressor().fit() download it) avoids also needing
a separate Prior Labs TABPFN_TOKEN. tabicl's checkpoint has no such gating, and
neither does tabpfn_v2's -- its HF repo (Prior-Labs/TabPFN-v2-reg) is public,
and v2 isn't one of the versions tabpfn's own model_loading gates behind Prior
Labs' license flow (only v2.5/v2.6/v3 are), so _prefetch_tabpfn_v2_checkpoint
downloads it anonymously, purely to bake it into the image layer.

Originally this fanned out per (embedding, method) only, looping over all 17
endpoints inside one container -- the wrong grain: a container's timeout had
to cover the *sum* of 17 fits (hard to size, since some endpoints finish in
seconds and others take over an hour), and any interruption (preemption,
timeout, a forced `--override` rerun) redid *all* 17, not just the one in
flight (this caused ChemFM-1B/ChemFM-3B/grover_large/grover_base/rmat_4M to
repeatedly stall during the RF grid-size fix). Per-endpoint containers make
each unit of work small, bounded, and independently resumable: a preemption
only costs one fit, `--override` is always safe, and Modal's own scheduler
parallelizes across all of it.

Before running for the first time, data/prepared and data/embedded need to
already be on the Modal Volume (same volume modal_embed.py writes to):

    modal volume put ml-embedding-benchmark data/prepared /prepared
    modal volume put ml-embedding-benchmark data/embedded /embedded

Run detached -- this fans out to (up to) ~57 embeddings x 3 methods x 17
endpoints ~= 2900 containers, and a non-detached `modal run` tears down the
whole app (killing every still-running container) the moment the local client
disconnects, same gotcha documented in scripts/modal_embed.py /
docs/openadmet_migration_status.md:

    modal run --detach scripts/modal_score_errors.py
    modal run --detach scripts/modal_score_errors.py --methods rf
    modal run --detach scripts/modal_score_errors.py --embeddings ECFP,coati --methods rf,ridge,knn
    modal run --detach scripts/modal_score_errors.py --endpoints reg_openadmet_pxr_pEC50
    modal run --detach scripts/modal_score_errors.py --embeddings ECFP --methods tabicl --endpoints reg_openadmet_expansionrx_LogD
    modal run --detach scripts/modal_score_errors.py --embeddings ECFP --methods tabpfn --endpoints reg_openadmet_expansionrx_LogD
    modal run --detach scripts/modal_score_errors.py --embeddings ECFP --methods tabpfn_v2 --endpoints reg_openadmet_expansionrx_LogD

Detached runs return immediately after spawning; check progress via
`modal app list` / `modal app logs <app-id>`, and Volume contents directly
(`modal volume ls ml-embedding-benchmark /errors`) as ground truth per the same
"modal app logs has been unreliable" note in docs/openadmet_migration_status.md.

After it completes, pull results back:

    modal volume get ml-embedding-benchmark /errors data/errors
"""

import os
import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VOLUME_NAME = "ml-embedding-benchmark"
MOUNT = pathlib.Path("/data")
DEFAULT_METHODS = ["rf", "ridge", "knn"]
GPU_METHODS = {"tabicl", "tabpfn", "tabpfn_v2"}
ALL_METHODS = DEFAULT_METHODS + ["tabicl", "tabpfn", "tabpfn_v2"]

# Modal secret (HF_TOKEN) authorizing access to tabpfn's gated default
# checkpoint -- see build_tabpfn_image().
HF_SECRET_NAME = "hf-ne"

# Checkpoint versions baked into the GPU image at build time (see
# build_tabicl_image) -- must match tabicl's TabICLRegressor default
# checkpoint_version so _load_model()'s local_files_only lookup hits the
# image's HF cache instead of re-downloading per container.
TABICL_REGRESSOR_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"

# Must match tabpfn's own default filename for the v3 regressor (see
# build_tabpfn_image / _prefetch_tabpfn_checkpoint) so a plain
# TabPFNRegressor().fit() at runtime resolves to this same cached path.
TABPFN_REGRESSOR_CHECKPOINT = "tabpfn-v3-regressor-v3_default.ckpt"

# Must match ModelSource.get_regressor_v2's default_filename (see
# build_tabpfn_v2_image / _prefetch_tabpfn_v2_checkpoint) so
# create_default_for_version(ModelVersion.V2) at runtime resolves to this
# same cached path.
TABPFN_V2_REGRESSOR_CHECKPOINT = "tabpfn-v2-regressor.ckpt"

app = modal.App("ml-embedding-benchmark-score")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    # Exact pins matching what's already validated locally against the actual
    # embedded/prepared joblib files (src/eval/supervised/regression.py's
    # GridSearchCV path, get_reg_models' rf/ridge/knn pipelines).
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "numpy==1.26.4",
        "pandas==2.3.3",
        "scikit-learn==1.9.0",
        "scikit-fingerprints==1.16.0",
        "joblib==1.4.2",
        # Pinned to match requirements.txt -- otherwise scipy/pyyaml/torch
        # resolve independently per image build, and an unpinned torch in
        # particular is a benchmark confound if the tabicl/tabpfn images end
        # up on different torch versions depending on when each was built.
        "scipy==1.13.1",
        "pyyaml==6.0.2",
        # src/common/types.py unconditionally imports torch (NumpyEncoder's
        # torch.Tensor handling) even though this scoring path never uses it --
        # same base dependency modal_embed.py's _install_base_requirements()
        # already carries for the same reason.
        "torch==2.5.1",
    )
    return (
        image.add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
        .add_local_file(
            PROJECT_ROOT / "scripts" / "compute_prediction_errors.py",
            remote_path="/project/scripts/compute_prediction_errors.py",
        )
    )


IMAGE = build_image()


def _prefetch_tabicl_checkpoint():
    # Bakes the pretrained checkpoint into the image layer (must be a
    # module-level function -- Modal's run_function() can't import a
    # closure) so containers don't each hit the Hugging Face Hub on cold
    # start -- both for speed and to avoid ~thousands of containers
    # redundantly pulling the same file.
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id="jingang/TabICL", filename=TABICL_REGRESSOR_CHECKPOINT)


def build_tabicl_image() -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "numpy==1.26.4",
        "pandas==2.3.3",
        "scikit-learn==1.9.0",
        "scikit-fingerprints==1.16.0",
        "joblib==1.4.2",
        # Pinned to match requirements.txt -- otherwise scipy/pyyaml/torch
        # resolve independently per image build, and an unpinned torch in
        # particular is a benchmark confound if the tabicl/tabpfn images end
        # up on different torch versions depending on when each was built.
        "scipy==1.13.1",
        "pyyaml==6.0.2",
        "torch==2.5.1",
        "tabicl==2.1.1",
        "einops==0.8.1",
    ).run_function(_prefetch_tabicl_checkpoint)
    return (
        image.add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
        .add_local_file(
            PROJECT_ROOT / "scripts" / "compute_prediction_errors.py",
            remote_path="/project/scripts/compute_prediction_errors.py",
        )
    )


TABICL_IMAGE = build_tabicl_image()


def _prefetch_tabpfn_checkpoint():
    # tabpfn's own download path (triggered by a plain TabPFNRegressor().fit())
    # gates v3 checkpoints behind Prior Labs' separate license-acceptance
    # server (a TABPFN_TOKEN from https://ux.priorlabs.ai, distinct from
    # HF_TOKEN) -- see tabpfn/browser_auth.py's ensure_license_accepted. That
    # check only runs when tabpfn's resolved cache path doesn't already exist
    # (model_loading.load_model_criterion_config: `if not path.exists():
    # download_model(...)`), so downloading the checkpoint ourselves straight
    # from the (merely HF-gated, not Prior-Labs-gated) HF repo into that same
    # cache path lets a plain TabPFNRegressor().fit() at runtime find it
    # already cached and skip the license flow entirely.
    from huggingface_hub import hf_hub_download
    from tabpfn.model_loading import get_cache_dir

    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / TABPFN_REGRESSOR_CHECKPOINT
    local_path = hf_hub_download(
        repo_id="Prior-Labs/tabpfn_3", filename=TABPFN_REGRESSOR_CHECKPOINT, local_dir=cache_dir,
    )
    os.rename(local_path, target)


def build_tabpfn_image() -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "numpy==1.26.4",
        "pandas==2.3.3",
        "scikit-learn==1.9.0",
        "scikit-fingerprints==1.16.0",
        "joblib==1.4.2",
        # Pinned to match requirements.txt -- otherwise scipy/pyyaml/torch
        # resolve independently per image build, and an unpinned torch in
        # particular is a benchmark confound if the tabicl/tabpfn images end
        # up on different torch versions depending on when each was built.
        "scipy==1.13.1",
        "pyyaml==6.0.2",
        "torch==2.5.1",
        "tabpfn==8.4.0",
    ).run_function(
        _prefetch_tabpfn_checkpoint,
        secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    )
    return (
        image.add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
        .add_local_file(
            PROJECT_ROOT / "scripts" / "compute_prediction_errors.py",
            remote_path="/project/scripts/compute_prediction_errors.py",
        )
    )


TABPFN_IMAGE = build_tabpfn_image()


def _prefetch_tabpfn_v2_checkpoint():
    # Unlike v3, v2's checkpoint repo (Prior-Labs/TabPFN-v2-reg) is public and
    # not gated behind Prior Labs' license flow, so this is a plain anonymous
    # download -- done here only to bake the checkpoint into the image layer,
    # same rationale as tabicl's prefetch above.
    from huggingface_hub import hf_hub_download
    from tabpfn.model_loading import get_cache_dir

    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / TABPFN_V2_REGRESSOR_CHECKPOINT
    local_path = hf_hub_download(
        repo_id="Prior-Labs/TabPFN-v2-reg", filename=TABPFN_V2_REGRESSOR_CHECKPOINT, local_dir=cache_dir,
    )
    os.rename(local_path, target)


def build_tabpfn_v2_image() -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "numpy==1.26.4",
        "pandas==2.3.3",
        "scikit-learn==1.9.0",
        "scikit-fingerprints==1.16.0",
        "joblib==1.4.2",
        # Pinned to match requirements.txt -- otherwise scipy/pyyaml/torch
        # resolve independently per image build, and an unpinned torch in
        # particular is a benchmark confound if the tabicl/tabpfn images end
        # up on different torch versions depending on when each was built.
        "scipy==1.13.1",
        "pyyaml==6.0.2",
        "torch==2.5.1",
        "tabpfn==8.4.0",
    ).run_function(_prefetch_tabpfn_v2_checkpoint)
    return (
        image.add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
        .add_local_file(
            PROJECT_ROOT / "scripts" / "compute_prediction_errors.py",
            remote_path="/project/scripts/compute_prediction_errors.py",
        )
    )


TABPFN_V2_IMAGE = build_tabpfn_v2_image()


def _ensure_symlink(target: str, link_path: str) -> None:
    # Modal reuses warm containers across multiple spawn()'d calls to the same
    # function -- a reused container already has this symlink from its prior
    # invocation, and a plain os.symlink() would crash with FileExistsError.
    if os.path.islink(link_path):
        return
    os.symlink(target, link_path, target_is_directory=True)


def _run_score(embedding: str, method: str, endpoint: str, n_jobs: int, override: bool) -> str:
    import subprocess

    # Ensures this container sees the latest committed Volume data even if it
    # was booted/warmed before an out-of-band `modal volume put` completed.
    volume.reload()

    project_root = "/project"
    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "errors").mkdir(parents=True, exist_ok=True)
    _ensure_symlink(str(MOUNT / "prepared"), f"{project_root}/data/prepared")
    _ensure_symlink(str(MOUNT / "embedded"), f"{project_root}/data/embedded")
    _ensure_symlink(str(MOUNT / "errors"), f"{project_root}/data/errors")

    cmd = [
        "python", "-u", "scripts/compute_prediction_errors.py",
        "--embedding", embedding,
        "--method", method,
        "--datasets", endpoint,
        "--n-jobs", str(n_jobs),
    ]
    if override:
        cmd.append("--override")

    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(
            f"compute_prediction_errors.py failed for embedding={embedding} method={method} "
            f"endpoint={endpoint} (exit {result.returncode})"
        )
    return f"ok  {embedding}/{method}/{endpoint}"


CPU = 8

score_fn = app.function(
    name="score_cpu",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    # A single endpoint's GridSearchCV fit -- worst observed so far (widest
    # embeddings x biggest datasets, e.g. rmat_4M/ChemFM-3B on pxr_pEC50 or
    # LogD) was ~1h40m. 3h leaves real margin without inheriting the old
    # per-container budget problem of needing to cover 17 fits' worth of time.
    timeout=60 * 60 * 3,
    cpu=CPU,
)(_run_score)

score_fn_gpu = app.function(
    name="score_gpu",
    image=TABICL_IMAGE,
    volumes={str(MOUNT): volume},
    timeout=60 * 60 * 3,
    # An A10G (24GB VRAM) hit repeated CUDA OOM retries fitting ECFP
    # (2048-dim, 5039 train rows) even with batch_size=1 (tabicl_head.py) --
    # it eventually succeeded via PyTorch's allocator backing off and
    # retrying, but that cost several extra minutes per fit. Some embeddings
    # here go up to 3072-dim, so L40S's 48GB gives real headroom instead of
    # relying on retries to bail us out at scale.
    gpu="L40S",
    cpu=4,
    # Generous relative to VRAM: TabICL's "auto" offload mode can still
    # spill large column-embedding tensors to host RAM on the widest
    # embeddings, and a 30GB CPU-only host OOM'd on this same data (see
    # tabicl_head.py) before that offload path was even exercised.
    memory=65536,
)(_run_score)

score_fn_gpu_tabpfn = app.function(
    name="score_gpu_tabpfn",
    image=TABPFN_IMAGE,
    volumes={str(MOUNT): volume},
    timeout=60 * 60 * 3,
    # Same tier as tabicl's score_fn_gpu to start -- TabPFN's own
    # memory_saving_mode="auto" (tabpfn_head.py) handles batching internally,
    # so this hasn't yet needed the same manual tuning tabicl did. Revisit if
    # OOMs show up on the widest embeddings.
    gpu="L40S",
    cpu=4,
    memory=65536,
)(_run_score)

score_fn_gpu_tabpfn_v2 = app.function(
    name="score_gpu_tabpfn_v2",
    image=TABPFN_V2_IMAGE,
    volumes={str(MOUNT): volume},
    timeout=60 * 60 * 3,
    # Same tier as score_fn_gpu_tabpfn -- v2 is the smaller/older model, so
    # this is a conservative starting point, not a measured requirement.
    gpu="L40S",
    cpu=4,
    memory=65536,
)(_run_score)

GPU_FUNCTIONS = {"tabicl": score_fn_gpu, "tabpfn": score_fn_gpu_tabpfn, "tabpfn_v2": score_fn_gpu_tabpfn_v2}


@app.local_entrypoint()
def main(
    embeddings: str = "",
    methods: str = ",".join(DEFAULT_METHODS),
    endpoints: str = "reg_openadmet_*",
    n_jobs: int = CPU,
    override: bool = False,
):
    """--embeddings: comma-separated embedding stems (default: every stem
    discovered under local data/embedded/*/ -- make sure that's populated
    locally, or pass an explicit list to score just a subset).
    --methods: comma-separated regression heads (default: rf,ridge,knn; also
    accepts 'tabicl', 'tabpfn', and 'tabpfn_v2', which each run on their own
    GPU-attached image/function).
    --endpoints: glob (without .yaml) over config/dataset/ endpoints, e.g.
    'reg_openadmet_pxr_pEC50' to smoke-test a single endpoint.
    --n-jobs: sklearn GridSearchCV parallelism per container (match --cpu).
    --override: recompute even if data/errors/<endpoint>/<embedding>__<method>.joblib
    already exists on the Volume -- always safe at this grain, since a rerun
    only ever touches the one file for this (embedding, method, endpoint)."""
    # Imported here, not at module scope: this file is also re-imported
    # *inside* the container to hydrate score_fn, where scripts/ (a
    # local-machine path) doesn't exist -- a module-level import crashed
    # every container on startup, which Modal kept retrying (looked like a
    # crash loop). main() only ever runs locally, so this is safe here.
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from compute_prediction_errors import discover_dataset_configs, discover_embeddings

    if embeddings:
        embedding_list = [e.strip() for e in embeddings.split(",") if e.strip()]
    else:
        embedding_list = discover_embeddings(str(PROJECT_ROOT / "data" / "embedded"))
    method_list = [m.strip() for m in methods.split(",") if m.strip()]
    endpoint_list = [
        c["name"] for c in discover_dataset_configs(str(PROJECT_ROOT / "config" / "dataset"), endpoints)
    ]

    unknown_methods = set(method_list) - set(ALL_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown method(s): {sorted(unknown_methods)}. Known: {ALL_METHODS}")

    total = len(embedding_list) * len(method_list) * len(endpoint_list)
    print(f"Fanning out {len(embedding_list)} embedding(s) x {len(method_list)} method(s) "
          f"x {len(endpoint_list)} endpoint(s) = {total} containers")

    calls = {}
    for embedding in embedding_list:
        for method in method_list:
            fn = GPU_FUNCTIONS.get(method, score_fn)
            for endpoint in endpoint_list:
                key = f"{embedding}/{method}/{endpoint}"
                calls[key] = fn.spawn(embedding, method, endpoint, n_jobs, override)

    succeeded = 0
    failed = []
    for key, call in calls.items():
        try:
            call.get()
            succeeded += 1
        except Exception as e:
            failed.append((key, str(e)))

    print(f"Done: {succeeded} succeeded, {len(failed)} failed")
    for key, err in failed:
        print(f"FAILED  {key}: {err}")
