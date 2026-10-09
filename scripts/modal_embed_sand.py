"""Run embed.py on Modal for just the `SAND` wrapper (both sand/sand_quantized
variants -- config/experiment/sand.yaml's own sweep already covers both, no
per-model fan-out needed), one container per dataset for parallelism.

    modal run scripts/modal_embed_sand.py
    modal run scripts/modal_embed_sand.py --datasets reg_openadmet_pxr_pEC50
    modal run scripts/modal_embed_sand.py --modal-gpu A10G
    modal run scripts/modal_embed_sand.py --modal-gpu none --modal-cpu 8

Mirrors scripts/modal_embed_unimol.py's structure (see that file for why this
is split out from modal_embed.py's multi-wrapper `_build_embed_functions()`).

Before running for the first time, upload data/prepared to the Volume (same
one modal_embed.py uses) if not already there:

    modal volume put ml-embedding-benchmark data/prepared /prepared

Pull results back locally with:

    modal volume get ml-embedding-benchmark /embedded data/embedded
"""

import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VOLUME_NAME = "ml-embedding-benchmark"
MOUNT = pathlib.Path("/data")

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

app = modal.App("ml-embedding-benchmark-sand")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/SAND"
    return (
        modal.Image.debian_slim(python_version="3.11")  # matches SAND's own .python-version
        # git: `pip install -e .` needs it for setuptools_scm's version
        # detection (sand's pyproject.toml declares no fallback_version, so it
        # requires the repo's own .git history at build time -- confirmed
        # `git describe` returns "v1.0.0" cleanly from this nested repo).
        # curl: init.sh downloads both checkpoints from GitHub releases.
        .apt_install("git", "curl")
        .pip_install(
            # embed.py / src/common/types.py need these regardless of wrapper;
            # everything else (torch, rdkit, pandas, numpy, ...) is declared by
            # sand's own pyproject.toml and installed by init.sh below -- no
            # separate pin needed here, unlike the legacy wrappers
            # modal_embed.py's _install_base_requirements supports.
            "hydra-core", "hydra-joblib-launcher", "joblib", "tqdm",
        )
        .add_local_dir(
            PROJECT_ROOT / "model_wrappers" / "SAND",
            remote_path=wrapper_dir,
            copy=True,  # bake into a layer so run_commands below is allowed
            # venv/ is a multi-GB local pyenv venv, irrelevant in the container
            # (which gets its own via pip_install). checkpoints/ is excluded
            # too -- init.sh re-downloads both fresh from GitHub at build time
            # rather than uploading ~640MB of local files.
            ignore=["venv", "venv/**", "checkpoints", "checkpoints/**"],
        )
        .run_commands(f"cd {wrapper_dir} && INSTALL_DEP=true bash -e init.sh")
        # repo mount, after run_commands -- Modal forbids run_commands after a
        # non-copy add_local_dir.
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config", remote_path="/project/config")
        .add_local_file(PROJECT_ROOT / "embed.py", remote_path="/project/embed.py")
    )


def _run_embed_sand(dataset_name: str, force: bool = False) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/SAND"

    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "embedded").mkdir(parents=True, exist_ok=True)
    os.symlink(str(MOUNT / "prepared"), f"{project_root}/data/prepared", target_is_directory=True)
    os.symlink(str(MOUNT / "embedded"), f"{project_root}/data/embedded", target_is_directory=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{project_root}:{wrapper_dir}"

    cmd = [
        "python", "-u", "embed.py",
        "--multirun",
        "+experiment=sand",
        f"+dataset={dataset_name}",
        f"++cache={'false' if force else 'true'}",
    ]

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"embed.py failed for dataset={dataset_name} (exit {result.returncode})")
    return f"ok  {dataset_name}"


IMAGE = build_image()

embed_sand = app.function(
    name="embed_sand",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    gpu="T4",
    cpu=4.0,
    timeout=60 * 60 * 3,
)(_run_embed_sand)


@app.local_entrypoint()
def main(datasets: str = ",".join(DATASETS), modal_gpu: str = "", modal_cpu: float = 0, force: bool = False):
    """--datasets: comma-separated dataset names (default: all 17). One Modal
    container per dataset, spawned concurrently -- both sand/sand_quantized
    variants run inside each container.
    --modal-gpu: override GPU type (default: T4). Pass "none" for CPU-only.
    --modal-cpu: override CPU core count (default: 4).
    --force: recompute and overwrite even if a cached .joblib already exists."""
    selected = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    fn = embed_sand
    options = {}
    if modal_gpu:
        options["gpu"] = None if modal_gpu.lower() == "none" else modal_gpu
    if modal_cpu:
        options["cpu"] = modal_cpu
    if options:
        fn = fn.with_options(**options)

    calls = {d: fn.spawn(d, force=force) for d in selected}
    for d, call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {d}: {e}")
