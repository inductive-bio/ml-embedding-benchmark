"""Run embed.py on Modal for the `KERMT` wrapper, one container per dataset.

    modal run scripts/modal_embed_kermt.py
    modal run scripts/modal_embed_kermt.py --datasets reg_openadmet_pxr_pEC50
    modal run scripts/modal_embed_kermt.py --modal-gpu A10G

KERMT produces a single "kermt" embedding: the paper's own canonical
fingerprint (`main.py fingerprint --fingerprint_source both`), i.e. the
concatenation of all four dual-view readouts into one 3200-d vector -- see
wrapper.py for why that's reimplemented via extract_embeddings_batch()
directly rather than shelling out to main.py.

KERMT's own environment.yml pins `cuik_molmaker>=0.3`, which NVIDIA only
publishes as a prebuilt conda package (conda-forge/nvidia channels) -- it has
no matching PyPI wheel, so this image is built with Modal's
`micromamba_install(spec_file=...)` directly against environment.yml rather
than a plain `pip_install` list (see model_wrappers/KERMT/init.sh for why the
usual pyenv+venv+pip local flow doesn't work for this wrapper). `hydra-core`/
`hydra-joblib-launcher` (needed by embed.py, not by KERMT itself) are added on
top with a plain pip_install, which runs inside the same activated conda env.

The released checkpoint (nvidia/NV-KERMT-70M-v2 on Hugging Face, pinned via
model_wrappers/KERMT/agent/config/released_model.json) is public and needs no
Modal credentials, so -- unlike chemeleon's runtime-mounted volume workaround
-- it's baked into the image at *build* time via `agent/scripts/
fetch_released_model.py`, same idea as SAND's build-time `curl`.

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

app = modal.App("ml-embedding-benchmark-kermt")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/KERMT"
    return (
        modal.Image.micromamba(python_version="3.11")  # matches KERMT's own environment.yml
        .micromamba_install(
            spec_file=str(PROJECT_ROOT / "model_wrappers" / "KERMT" / "environment.yml"),
            channels=["conda-forge", "nvidia", "nodefaults"],
            # conda's solver needs the `__cuda` virtual package for pytorch-gpu's
            # dependency solve; attaching a real GPU to the builder (`gpu=`)
            # didn't make mamba detect it, so force it explicitly -- standard
            # for building CUDA-dependent conda envs without a GPU present
            # at build time.
            env={"CONDA_OVERRIDE_CUDA": "12.6"},
        )
        .pip_install("hydra-core", "hydra-joblib-launcher", "joblib", "tqdm")
        .add_local_dir(
            PROJECT_ROOT / "model_wrappers" / "KERMT",
            remote_path=wrapper_dir,
            ignore=["venv", "venv/**", ".git", ".git/**", "runs", "runs/**", "model", "model/**"],
            copy=True,
        )
        .run_commands(
            # The conda/micromamba env's Python can't find a valid CA bundle by
            # default (SSL_CERT_FILE unset), so huggingface_hub's HTTPS download
            # fails cert verification -- point it at certifi's bundle explicitly.
            f"cd {wrapper_dir} && python -m pip install --quiet certifi && "
            f"SSL_CERT_FILE=$(python -m certifi) REQUESTS_CA_BUNDLE=$(python -m certifi) "
            f"python agent/scripts/fetch_released_model.py --out checkpoints"
        )
        # repo mount, after run_commands -- Modal forbids run_commands after a
        # non-copy add_local_dir.
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config", remote_path="/project/config")
        .add_local_file(PROJECT_ROOT / "embed.py", remote_path="/project/embed.py")
    )


def _run_embed_kermt(dataset_name: str, force: bool = False) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/KERMT"

    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "embedded").mkdir(parents=True, exist_ok=True)
    # Modal warm-starts/reuses containers across multiple .spawn() calls -- a
    # reused container already has these symlinks from its prior call, so
    # guard with lexists (not exists, which follows and can misbehave on a
    # dangling symlink) like every other recent modal_embed_*.py script.
    if not os.path.lexists(f"{project_root}/data/prepared"):
        os.symlink(str(MOUNT / "prepared"), f"{project_root}/data/prepared", target_is_directory=True)
    if not os.path.lexists(f"{project_root}/data/embedded"):
        os.symlink(str(MOUNT / "embedded"), f"{project_root}/data/embedded", target_is_directory=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{project_root}:{wrapper_dir}"

    cmd = [
        "python", "-u", "embed.py",
        "+experiment=kermt",
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

embed_kermt = app.function(
    name="embed_kermt",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    gpu="T4",
    cpu=4.0,
    timeout=60 * 30,
)(_run_embed_kermt)


@app.local_entrypoint()
def main(
    datasets: str = ",".join(DATASETS),
    modal_gpu: str = "",
    modal_cpu: float = 0,
    force: bool = False,
):
    """--datasets: comma-separated dataset names (default: all 17). One Modal
    container per dataset, spawned concurrently.
    --modal-gpu: override GPU type (default: T4). Pass "none" for CPU-only.
    --modal-cpu: override CPU core count (default: 4).
    --force: recompute and overwrite even if a cached .joblib already exists."""
    selected = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    fn = embed_kermt
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
