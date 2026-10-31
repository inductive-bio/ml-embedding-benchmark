"""Run embed.py on Modal for the `chemeleon` wrapper, one container per
(model variant, dataset) pair, for maximum fan-out (unlike modal_embed_sand.py,
which runs all of a wrapper's variants inside one container per dataset).

    modal run scripts/modal_embed_chemeleon.py
    modal run scripts/modal_embed_chemeleon.py --models chemeleon_original --datasets reg_openadmet_pxr_pEC50
    modal run scripts/modal_embed_chemeleon.py --modal-gpu A10G

The published checkpoint is downloaded into the image at build time and
symlinked into model_wrappers/chemeleon/checkpoints/ before invoking embed.py.

Before running for the first time, upload data/prepared to the Volume (same
one modal_embed.py uses) if not already there:

    modal volume put ml-embedding-benchmark data/prepared /prepared

Pull results back locally with:

    modal volume get ml-embedding-benchmark /embedded data/embedded

(Note: that bulk `get` has not reliably merged new files into already
-populated local dataset directories in past sessions -- verify per-file
after pulling and re-fetch anything missing individually.)
"""

import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VOLUME_NAME = "ml-embedding-benchmark"
MOUNT = pathlib.Path("/data")
CHECKPOINT_DIR = pathlib.Path("/checkpoints")

DATASETS = [
    "reg_openadmet_expansionrx_Log_Caco_ER",
    "reg_openadmet_expansionrx_Log_Caco_Papp_AB",
    "reg_openadmet_expansionrx_LogD",
    "reg_openadmet_expansionrx_Log_HLM_CLint",
    "reg_openadmet_expansionrx_Log_MLM_CLint",
    "reg_openadmet_expansionrx_Log_Mouse_BPB",
    "reg_openadmet_expansionrx_Log_Mouse_MPB",
    "reg_openadmet_expansionrx_Log_Mouse_PPB",
    "reg_openadmet_expansionrx_KSOL",
    "reg_openadmet_polaris_admet_HLM",
    "reg_openadmet_polaris_admet_KSOL",
    "reg_openadmet_polaris_admet_LogD",
    "reg_openadmet_polaris_admet_MDR1_MDCKII",
    "reg_openadmet_polaris_admet_MLM",
    "reg_openadmet_polaris_potency_pIC50_MERS_CoV_Mpro",
    "reg_openadmet_polaris_potency_pIC50_SARS_CoV_2_Mpro",
    "reg_openadmet_pxr_pEC50",
]

# model_name -> (download URL, local filename expected by
# model_wrappers/chemeleon/wrapper.py's and finetune.py's _VARIANTS).
CHECKPOINTS = {
    "chemeleon_original": ("https://zenodo.org/records/15460715/files/chemeleon_mp.pt", "chemeleon_mp.pt"),
}

app = modal.App("ml-embedding-benchmark-chemeleon")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/chemeleon"
    return (
        modal.Image.debian_slim(python_version="3.11")  # matches chemeleon's own .python-version
        .pip_install(
            "hydra-core", "hydra-joblib-launcher", "joblib", "tqdm",
            # Pinned for the same reason as SAND/local init.sh: chemprop's own
            # torch>=2.1 dependency is unpinned and would otherwise resolve a
            # wheel needing a newer CUDA driver than some GPU nodes have.
            "torch==2.5.1", "chemprop==2.2.4",
        )
        .apt_install("curl")
        .run_commands(
            f"mkdir -p {CHECKPOINT_DIR}",
            *(f"curl -fL -o {CHECKPOINT_DIR}/{name} {url}" for url, name in CHECKPOINTS.values()),
        )
        .add_local_dir(
            PROJECT_ROOT / "model_wrappers" / "chemeleon",
            remote_path=wrapper_dir,
            ignore=["venv", "venv/**", "checkpoints", "checkpoints/**"],
        )
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config", remote_path="/project/config")
        .add_local_file(PROJECT_ROOT / "embed.py", remote_path="/project/embed.py")
    )


def _run_embed_chemeleon(model_name: str, dataset_name: str, force: bool = False) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/chemeleon"

    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "embedded").mkdir(parents=True, exist_ok=True)
    # Modal warm-starts containers across multiple .spawn() calls (confirmed
    # live: the first ~16 of 119 calls in an earlier run succeeded, then every later one failed
    # with `FileExistsError: /project/data/prepared` from a reused
    # container's prior symlink) -- guard every symlink the same way the
    # checkpoint one below already is.
    if not os.path.exists(f"{project_root}/data/prepared"):
        os.symlink(str(MOUNT / "prepared"), f"{project_root}/data/prepared", target_is_directory=True)
    if not os.path.exists(f"{project_root}/data/embedded"):
        os.symlink(str(MOUNT / "embedded"), f"{project_root}/data/embedded", target_is_directory=True)

    _, local_name = CHECKPOINTS[model_name]
    os.makedirs(f"{wrapper_dir}/checkpoints", exist_ok=True)
    checkpoint_dst = f"{wrapper_dir}/checkpoints/{local_name}"
    if not os.path.exists(checkpoint_dst):
        os.symlink(str(CHECKPOINT_DIR / local_name), checkpoint_dst)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{project_root}:{wrapper_dir}"

    cmd = [
        "python", "-u", "embed.py",
        f"+dataset={dataset_name}",
        f"+model_name={model_name}",
        f"++cache={'false' if force else 'true'}",
    ]

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"embed.py failed for model={model_name} dataset={dataset_name} (exit {result.returncode})")
    return f"ok  {model_name}  {dataset_name}"


IMAGE = build_image()

embed_chemeleon = app.function(
    name="embed_chemeleon",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    gpu="T4",
    cpu=4.0,
    timeout=60 * 30,  # one (model, dataset) pair per call -- far shorter than a multi-variant-per-container run
)(_run_embed_chemeleon)


@app.local_entrypoint()
def main(
    models: str = ",".join(CHECKPOINTS),
    datasets: str = ",".join(DATASETS),
    modal_gpu: str = "",
    modal_cpu: float = 0,
    force: bool = False,
):
    """--models: comma-separated model_name values (default: all).
    --datasets: comma-separated dataset names (default: all 17).
    Spawns one Modal container per (model, dataset) pair.
    --modal-gpu: override GPU type (default: T4). Pass "none" for CPU-only.
    --modal-cpu: override CPU core count (default: 4).
    --force: recompute and overwrite even if a cached .joblib already exists."""
    selected_models = [m.strip() for m in models.split(",") if m.strip()]
    unknown_models = set(selected_models) - set(CHECKPOINTS)
    if unknown_models:
        raise ValueError(f"Unknown model(s): {sorted(unknown_models)}. Known: {sorted(CHECKPOINTS)}")

    selected_datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown_datasets = set(selected_datasets) - set(DATASETS)
    if unknown_datasets:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown_datasets)}. Known: {DATASETS}")

    fn = embed_chemeleon
    options = {}
    if modal_gpu:
        options["gpu"] = None if modal_gpu.lower() == "none" else modal_gpu
    if modal_cpu:
        options["cpu"] = modal_cpu
    if options:
        fn = fn.with_options(**options)

    calls = {
        (m, d): fn.spawn(m, d, force=force)
        for m in selected_models
        for d in selected_datasets
    }
    for (m, d), call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {m}  {d}: {e}")
