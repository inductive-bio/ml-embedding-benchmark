"""Fine-tune the KERMT pretrained checkpoint end-to-end on Modal, one
container per dataset.

    modal run scripts/modal_finetune_kermt.py --datasets reg_openadmet_pxr_pEC50
    modal run scripts/modal_finetune_kermt.py --modal-gpu A100-80GB

Mirrors scripts/modal_embed_kermt.py's image (micromamba built directly from
environment.yml -- see that script for why: cuik-molmaker>=0.3 is conda-only,
no PyPI wheel). KERMT's own README recommends >=32GB VRAM for fine-tuning
(joint encoder+FFN training, unlike the lighter embed-only forward pass) --
default GPU tier here is bumped to A100-80GB accordingly, well above the T4
every other wrapper's finetune script uses; --modal-gpu overrides if a
smaller/larger tier turns out to be enough once a real run is tried.

Before running for the first time, data/prepared must already be on the
Volume (confirmed via `modal volume ls ml-embedding-benchmark /prepared`).
Pull results back locally with:

    modal volume get ml-embedding-benchmark /finetune_errors data/finetune_errors
    modal volume get ml-embedding-benchmark /finetune_checkpoints data/finetune_checkpoints
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

app = modal.App("ml-embedding-benchmark-kermt-finetune")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/KERMT"
    return (
        modal.Image.micromamba(python_version="3.11")  # matches KERMT's own environment.yml
        .micromamba_install(
            spec_file=str(PROJECT_ROOT / "model_wrappers" / "KERMT" / "environment.yml"),
            channels=["conda-forge", "nvidia", "nodefaults"],
            # conda's solver needs the `__cuda` virtual package for pytorch-gpu's
            # own dependency solve, which mamba doesn't detect just from a GPU
            # attached to the builder -- force it explicitly (see
            # modal_embed_kermt.py, where this was worked out live).
            env={"CONDA_OVERRIDE_CUDA": "12.6"},
        )
        .pip_install("joblib", "tqdm", "pyyaml")
        .add_local_dir(
            PROJECT_ROOT / "model_wrappers" / "KERMT",
            remote_path=wrapper_dir,
            ignore=["venv", "venv/**", ".git", ".git/**", "runs", "runs/**", "model", "model/**"],
            copy=True,
        )
        .run_commands(
            # See modal_embed_kermt.py -- the conda env's Python has no CA
            # bundle configured by default, breaking huggingface_hub's HTTPS
            # download's cert verification.
            f"cd {wrapper_dir} && python -m pip install --quiet certifi && "
            f"SSL_CERT_FILE=$(python -m certifi) REQUESTS_CA_BUNDLE=$(python -m certifi) "
            f"python agent/scripts/fetch_released_model.py --out checkpoints"
        )
        # repo mount, after run_commands -- Modal forbids run_commands after a
        # non-copy add_local_dir.
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
    )


def _run_finetune_kermt(
    dataset_name: str,
    epochs: int,
    run_tag: str = "",
    force: bool = False,
) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/KERMT"

    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "finetune_errors").mkdir(parents=True, exist_ok=True)
    (MOUNT / "finetune_checkpoints").mkdir(parents=True, exist_ok=True)
    log_dir = MOUNT / "finetune_logs" / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)
    # Modal reuses a warm container across multiple sequential inputs -- a
    # symlink from an earlier input on the same container already exists by
    # the time a later input runs there, so this must be idempotent.
    for src, dst in [
        (MOUNT / "prepared", f"{project_root}/data/prepared"),
        (MOUNT / "finetune_errors", f"{project_root}/data/finetune_errors"),
        (MOUNT / "finetune_checkpoints", f"{project_root}/data/finetune_checkpoints"),
    ]:
        if not os.path.lexists(dst):
            os.symlink(str(src), dst, target_is_directory=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = project_root

    cmd = [
        "python", "-u", f"{wrapper_dir}/finetune.py",
        "--dataset", dataset_name,
        "--config-dir", f"{project_root}/config/dataset",
        "--epochs", str(epochs),
    ]
    if run_tag:
        cmd += ["--run-tag", run_tag]
    if force:
        cmd.append("--override")

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    # Written to the Volume rather than relied on via `print()` -- KERMT's own
    # training loop is tqdm/logger-heavy and can flood/truncate Modal's
    # captured log before a real traceback ever shows up there (same issue
    # docs/openadmet_migration_status.md hit with ChemFM's embedding run).
    log_name = f"kermt_{run_tag}" if run_tag else "kermt"
    log_path = log_dir / f"{log_name}.log"
    log_path.write_text(f"=== stdout ===\n{result.stdout}\n=== stderr ===\n{result.stderr}\n")
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(
            f"finetune.py failed for dataset={dataset_name} run_tag={run_tag!r} (exit {result.returncode}) "
            f"-- see /finetune_logs/{dataset_name}/{log_name}.log on the Volume"
        )
    return f"ok  {dataset_name}  {run_tag}"


IMAGE = build_image()

finetune_kermt = app.function(
    name="finetune_kermt",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    gpu="A100-80GB",
    cpu=4.0,
    timeout=60 * 60 * 6,  # joint encoder+FFN training is far slower than a frozen-embedding forward pass
)(_run_finetune_kermt)


@app.local_entrypoint()
def main(
    datasets: str = ",".join(DATASETS),
    epochs: int = 30,  # agent/config/defaults_finetune.json's own default
    run_tag: str = "",
    modal_gpu: str = "",
    force: bool = False,
):
    """--datasets: comma-separated dataset names (default: all 17).
    --epochs: KERMT finetune epochs (default 30, matching agent/config/defaults_finetune.json).
    --run-tag: optional label (e.g. 'epochs50') for a hyperparameter-sweep
    run, so it's saved alongside the default config's result under
    data/finetune_errors/<dataset>/kermt_<run_tag>__finetuned.joblib instead
    of overwriting it.
    --modal-gpu: override GPU type (default: A100-80GB, per KERMT's own
    >=32GB VRAM recommendation for fine-tuning).
    --force: recompute and overwrite even if a cached result already exists."""
    selected = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    fn = finetune_kermt
    if modal_gpu:
        fn = fn.with_options(gpu=None if modal_gpu.lower() == "none" else modal_gpu)

    calls = {d: fn.spawn(d, epochs, run_tag=run_tag, force=force) for d in selected}
    for d, call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {d}: {e}")
