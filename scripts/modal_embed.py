"""Run embed.py on Modal, once per model wrapper, for the 17 new OpenADMET
regression datasets (config/dataset/reg_openadmet_*.yaml).

    modal run scripts/modal_embed.py --wrappers fingerprints
    modal run scripts/modal_embed.py --wrappers fingerprints,mol2vec,unimol
    modal run scripts/modal_embed.py                       # all default (automated) wrappers
    modal run scripts/modal_embed.py --wrappers huggingface --modal-gpu A10G

Before running this for the first time, upload data/prepared/reg_openadmet_*.joblib
(built locally by `python download.py --multirun '+dataset=glob(reg_openadmet_*)'`)
to the Modal Volume. Note the destination is `/prepared`, not `/data/prepared` --
this script mounts the volume's root at container path /data (MOUNT below), so a
volume-internal path of `/prepared` is what ends up at `/data/prepared` inside
the container:

    modal volume put ml-embedding-benchmark data/prepared /prepared

After a run, pull results back locally with:

    modal volume get ml-embedding-benchmark /embedded data/embedded

Mirrors embed_wrapper.sh's local flow (per-wrapper venv -> that wrapper's own
init.sh -> `python embed.py --multirun +experiment=<name>`), except each
wrapper's init.sh runs once at Modal *image build* time (baked into a layer)
instead of at every container start, and data/prepared (read) + data/embedded
(write) live on a Modal Volume instead of local disk. Idempotency comes from
this repo's own `is_already_embedded()` + `cache: true` (see
src/embedding/embedding.py) -- as long as data/embedded persists across runs
on the Volume, re-running is safe and just fills in whatever's missing.

Explicitly excluded, two different reasons:
- `cddd`, `chemformer`, `simson`, and the `huggingface` wrapper's
  `selformer`/`selformer_lite` model variants: need an interactive Google
  Drive/Box/Figshare download a container can't do non-interactively (see
  docs/data_sources/ and the approved plan for this work). huggingface's
  other ~11 variants pull from the HF Hub at runtime and are included.
- `graphfp`, `graphmvp`, `mol-r`, `molbert`: pinned to exact old
  torch/dgl/transformers versions with no wheel for Python >=3.10, which
  Modal requires (see _UNFIXED_LEGACY_PYTHON_WRAPPERS below).
"""

import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VOLUME_NAME = "ml-embedding-benchmark"
MOUNT = pathlib.Path("/data")
DATASET_GLOB = "glob(reg_openadmet_*)"

app = modal.App("ml-embedding-benchmark-embed")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ---------------------------------------------------------------------------
# Per-wrapper metadata
#   hydra_experiment : matches that wrapper's init.sh `export HYDRA_EXPERIMENT=...`
#   pythonpath_extra : matches any *extra* PYTHONPATH entry that wrapper's own
#                       init.sh adds beyond the repo root + wrapper dir (which
#                       embed_wrapper.sh always adds -- replicated below for
#                       every wrapper regardless)
#   model_glob       : Hydra override for `+model=glob(...)` -- defaults to
#                       that wrapper's own config/experiment/<name>.yaml sweep
#                       ("*") unless a variant needs excluding
# ---------------------------------------------------------------------------

WRAPPERS = {
    "clamp": dict(hydra_experiment="clamp", pythonpath_extra=[]),
    "coati": dict(hydra_experiment="coati", pythonpath_extra=["coati_repo"]),
    "fingerprints": dict(hydra_experiment="fingerprints", pythonpath_extra=[]),
    "gem": dict(hydra_experiment="gem", pythonpath_extra=["gem_repo"]),
    "graphfp": dict(hydra_experiment="graphfp", pythonpath_extra=["graphfp_repo/mol"]),
    "graphmvp": dict(hydra_experiment="graphmvp", pythonpath_extra=[]),
    "huggingface": dict(
        hydra_experiment="huggingface",
        pythonpath_extra=[],
        model_glob="glob(huggingface*,exclude=[*selformer*])",  # skip selformer/selformer_lite -- manual GDrive weights
    ),
    "huggingmolecules": dict(hydra_experiment="huggingmolecules", pythonpath_extra=[]),
    "mol2vec": dict(hydra_experiment="mol2vec", pythonpath_extra=[]),
    "mol-r": dict(hydra_experiment="mol_r", pythonpath_extra=["mol_r_repo/src"]),
    "molbert": dict(hydra_experiment="molbert", pythonpath_extra=["molbert_repo"]),
    "unimol": dict(hydra_experiment="unimol", pythonpath_extra=[]),
}

# T4 by default for anything torch-based and not explicitly CPU-pinned.
# graphfp/graphmvp pin a CPU-only torch wheel (no GPU benefit); gem's
# paddlepaddle pin (requirements.txt: "paddlepaddle>=2.0.0", not the -gpu
# variant) is also CPU-only as currently declared. fingerprints/mol2vec have
# no DL framework at inference time.
GPU_FOR = {
    "clamp": "T4",
    "coati": "T4",
    "huggingface": "T4",
    "huggingmolecules": "T4",
    "unimol": "T4",
}

# graphfp/graphmvp/mol-r/molbert stay in WRAPPERS (so requesting them by name
# gives a clear error via build_image) but are excluded from the default set
# -- see _UNFIXED_LEGACY_PYTHON_WRAPPERS below.
DEFAULT_WRAPPERS = sorted(set(WRAPPERS) - {"graphfp", "graphmvp", "mol-r", "molbert"})


# ---------------------------------------------------------------------------
# Image construction -- one per wrapper, each running that wrapper's *own*
# init.sh (already pinned to commit SHAs where it had floating git deps, see
# model_wrappers/{clamp,graphmvp,huggingmolecules,mol2vec}) at build time.
# ---------------------------------------------------------------------------


def _repo_mount(image: modal.Image) -> modal.Image:
    """Mount the parts of the repo embed.py needs, after all build-time
    `run_commands` are done (Modal forbids run_commands after a non-copy
    add_local_dir)."""
    return (
        image.add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config", remote_path="/project/config")
        .add_local_file(PROJECT_ROOT / "embed.py", remote_path="/project/embed.py")
    )


def _install_base_requirements(image: modal.Image) -> modal.Image:
    """Mirrors embed_wrapper.sh's `pip install -r ../../base_requirements.txt`
    step, run before every wrapper's own init.sh. numpy<2 avoids the
    pandas-wheel ABI break that broke unimol's `pandas<2` pin (numpy 2.x is
    ABI-incompatible with pandas wheels built against numpy 1.x). rdkit is
    added here because coati/huggingmolecules import it directly without
    declaring it as a dependency. setuptools<81 avoids `ModuleNotFoundError:
    No module named 'pkg_resources'` (recent setuptools no longer vendors it,
    but clamp_repo's and pytdc's setup.py still `import pkg_resources`
    directly). git is needed at build time for the `pip install
    git+https://...` deps (clamp, coati, huggingmolecules, mol2vec) --
    debian_slim doesn't include it by default."""
    image = image.apt_install("git")
    return image.pip_install(
        "hydra-core", "hydra-joblib-launcher", "torch", "joblib", "tqdm",
        "pandas", "numpy<2.0.0,>=1.26.4", "rdkit", "setuptools<81",
    )


def _wrapper_base_and_init(
    wrapper: str,
    base: modal.Image,
    apt_packages: list[str] = (),
    post_init_pip: list[str] = (),
) -> modal.Image:
    """Copy model_wrappers/<wrapper> into the image and run its init.sh with
    INSTALL_DEP=true, exactly as embed_wrapper.sh does locally -- just once,
    at build time, instead of on every container start."""
    wrapper_dir = f"/project/model_wrappers/{wrapper}"
    image = _install_base_requirements(base)
    if apt_packages:
        image = image.apt_install(*apt_packages)
    image = image.add_local_dir(
        PROJECT_ROOT / "model_wrappers" / wrapper,
        remote_path=wrapper_dir,
        copy=True,  # bake into a layer so run_commands below is allowed
        # Local venv/ dirs (and huggingface's venv_molformer/) are multi-GB,
        # irrelevant in the container (which gets its own via pip_install),
        # and slow/fatal to upload -- same exclusion as modal_embed_unimol.py/
        # modal_embed_selformer.py, generalized to any venv-like dir name.
        ignore=["venv", "venv/**", "venv_*", "venv_*/**"],
    )
    # `-e` (errexit) is passed to bash invoking init.sh, not added inside
    # init.sh itself -- none of these scripts declare `set -e`, so a failing
    # pip install would otherwise be silently swallowed (script continues,
    # exit code reflects whatever command ran last). This silently baked
    # broken images for several wrappers before being added.
    image = image.run_commands(f"cd {wrapper_dir} && INSTALL_DEP=true bash -e init.sh")
    if post_init_pip:
        # Some wrappers' init.sh pulls a dependency past a pin
        # _install_base_requirements already asserted (e.g. unimol's `pip
        # install unimol_tools --upgrade` pulls numpy back past <2.0) --
        # re-assert after the fact, same as the local run recipe does.
        image = image.pip_install(*post_init_pip)
    return image



# Modal's client library refuses to import under Python <3.10
# (`RuntimeError: This version of Modal requires at least Python 3.10`,
# raised inside the container itself, so an arbitrary from_registry base
# doesn't help). Wrappers whose local `.python-version` is <3.10
# (fingerprints/clamp/coati/huggingmolecules=3.9, gem=3.8) are bumped to 3.11
# for the Modal image only; local pyenv venvs are untouched. None pin a
# Python-ABI-specific wheel in init.sh, so this should work, but only
# fingerprints has been validated live -- watch the first run of each closely.
MODAL_PYTHON_VERSION = {
    "fingerprints": "3.11",
    "mol2vec": "3.11",
    "clamp": "3.11",
    "coati": "3.11",
    "huggingmolecules": "3.11",
    "gem": "3.11",
    "unimol": "3.10",  # matches local pin already, no bump needed
    "huggingface": "3.11",  # no local pin at all; this is the chosen default
}

# graphfp/graphmvp are pinned to Python 3.7 locally and their init.sh
# hardcodes cp37m-tagged wheel URLs (torch_scatter/torch_sparse); mol-r pins
# torch==1.8.1/dgl==0.6.1 with the same problem. molbert_repo/setup.py pins
# torch==1.4.0 and transformers==3.5.1 directly (confirmed live once the
# `bash -e` fix below started surfacing real pip failures instead of
# swallowing them). None of these four have a wheel for Python >=3.10, which
# Modal requires, so they're deliberately excluded from DEFAULT_WRAPPERS
# until fixed -- e.g. a dual-interpreter image (modern Python for Modal's
# runtime, pinned legacy Python via pyenv invoked by subprocess, mirroring
# embed_wrapper.sh) or, for molbert specifically, relaxing its pins and
# verifying embeddings are unchanged under a newer torch/transformers.
_UNFIXED_LEGACY_PYTHON_WRAPPERS = {"graphfp", "graphmvp", "mol-r", "molbert"}


# Per-wrapper build overrides beyond the shared _wrapper_base_and_init path.
# unimol needs both, confirmed live in scripts/modal_embed_unimol.py:
# - apt packages: unimol_tools imports rdkit.Chem.PandasTools -> ... ->
#   rdkit.Chem.Draw at import time (side effect only), which needs X11 libs
#   debian_slim doesn't ship (`ImportError: libXrender.so.1: ...`).
# - post_init_pip: unimol_tools' own `pip install --upgrade` in init.sh pulls
#   numpy back past the <2.0 pin, same ABI break documented in
#   _install_base_requirements.
_APT_PACKAGES_FOR = {
    "unimol": ["libxrender1", "libxext6", "libsm6"],
}
_POST_INIT_PIP_FOR = {
    "unimol": ["numpy<2.0.0"],
}


def build_image(wrapper: str) -> modal.Image:
    if wrapper not in MODAL_PYTHON_VERSION:
        raise ValueError(
            f"Unknown wrapper or no Modal Python version decided yet: {wrapper} "
            f"(see _UNFIXED_LEGACY_PYTHON_WRAPPERS)"
        )

    base = modal.Image.debian_slim(python_version=MODAL_PYTHON_VERSION[wrapper])
    image = _wrapper_base_and_init(
        wrapper,
        base,
        apt_packages=_APT_PACKAGES_FOR.get(wrapper, []),
        post_init_pip=_POST_INIT_PIP_FOR.get(wrapper, []),
    )

    return _repo_mount(image)


# ---------------------------------------------------------------------------
# The actual unit of work, run inside the container. Defined once; wrapped
# with a different `@app.function(image=...)` per wrapper below, since a
# single Modal function's image can't be swapped per-call.
# ---------------------------------------------------------------------------


def _run_embed(wrapper: str, dataset_glob: str, model_glob: str, hydra_experiment: str, pythonpath_extra: list) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/{wrapper}"

    # Pull data/prepared from the Volume into the mounted project layout, and
    # write data/embedded back to the Volume -- embed.py itself is oblivious
    # to Modal; it just sees local paths under the container's cwd.
    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "embedded").mkdir(parents=True, exist_ok=True)
    os.symlink(str(MOUNT / "prepared"), f"{project_root}/data/prepared", target_is_directory=True)
    os.symlink(str(MOUNT / "embedded"), f"{project_root}/data/embedded", target_is_directory=True)

    pythonpath_entries = [project_root, wrapper_dir] + [f"{wrapper_dir}/{p}" for p in pythonpath_extra]
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)

    # Both `dataset` and `model` are only ever defined via `hydra.sweeper.params`
    # (see config/embed.yaml and config/experiment/*.yaml), never via a
    # `defaults:` list entry -- so, same as the local `+dataset=glob(...)`
    # override already verified to work against download.py's analogous
    # config/downloader.yaml, these need the `+` prefix here too.
    cmd = [
        "python", "-u", "embed.py",
        "--multirun",
        f"+experiment={hydra_experiment}",
        f"+dataset={dataset_glob}",
        "++cache=true",
    ]
    if model_glob:
        cmd.append(f"+model={model_glob}")

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"embed.py failed for wrapper={wrapper} (exit {result.returncode})")
    return f"ok  {wrapper}"


def _build_embed_functions() -> dict:
    # Each wrapper gets its own registered Modal function (distinct `name=`,
    # since app.function() otherwise registers by the wrapped callable's
    # __name__ -- reusing _run_embed unnamed for all 12 would silently
    # collapse them into one registration, last-wrapper-wins).
    fns = {}
    for wrapper, meta in WRAPPERS.items():
        if wrapper in _UNFIXED_LEGACY_PYTHON_WRAPPERS:
            continue  # can't build yet -- see build_image()
        image = build_image(wrapper)
        safe_name = wrapper.replace("-", "_")
        fns[wrapper] = app.function(
            name=f"embed_{safe_name}",
            image=image,
            volumes={str(MOUNT): volume},
            gpu=GPU_FOR.get(wrapper),
            timeout=60 * 60 * 6,
        )(_run_embed)
    return fns


EMBED_FUNCTIONS = _build_embed_functions()


@app.local_entrypoint()
def main(wrappers: str = ",".join(DEFAULT_WRAPPERS), modal_gpu: str = "", dataset_glob: str = DATASET_GLOB, model_glob: str = ""):
    """--wrappers: comma-separated wrapper names (default: all included wrappers).
    --modal-gpu: override the GPU tier for every selected wrapper (default: T4
    for the wrappers in GPU_FOR, CPU-only for the rest).
    --dataset-glob: override the dataset glob (default: all 17 OpenADMET
    datasets) -- e.g. `glob(reg_openadmet_pxr_pEC50)` to smoke-test a single
    dataset instead of waiting on all 17, cheaper for diagnosing a wrapper
    that's failing outright.
    --model-glob: override the per-wrapper default model glob (e.g. restrict
    huggingface to one variant for a smoke test) -- applies to every selected
    wrapper, so only pass this when selecting a single wrapper."""
    selected = [w.strip() for w in wrappers.split(",") if w.strip()]
    unknown = set(selected) - set(WRAPPERS)
    if unknown:
        raise ValueError(f"Unknown wrapper(s): {sorted(unknown)}. Known: {DEFAULT_WRAPPERS}")
    unfixed = set(selected) & _UNFIXED_LEGACY_PYTHON_WRAPPERS
    if unfixed:
        raise ValueError(
            f"{sorted(unfixed)} can't run on Modal yet (Python <3.10, see "
            "_UNFIXED_LEGACY_PYTHON_WRAPPERS in build_image())."
        )

    calls = {}
    for wrapper in selected:
        meta = WRAPPERS[wrapper]
        fn = EMBED_FUNCTIONS[wrapper]
        if modal_gpu:
            fn = fn.with_options(gpu=modal_gpu)
        calls[wrapper] = fn.spawn(
            wrapper,
            dataset_glob,
            model_glob or meta.get("model_glob"),
            meta["hydra_experiment"],
            meta["pythonpath_extra"],
        )

    for wrapper, call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {wrapper}: {e}")
