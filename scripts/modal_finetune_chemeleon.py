"""Fine-tune a pretrained Chemeleon encoder end-to-end (encoder + fresh
single-task regression head) on Modal, one container per (model variant,
dataset) pair -- see docs/e2e_chemprop_chemeleon_plan.md.

    modal run scripts/modal_finetune_chemeleon.py --models chemeleon_original --datasets reg_openadmet_polaris_admet_KSOL
    modal run scripts/modal_finetune_chemeleon.py --modal-gpu T4

Checkpoints are not baked into the image at build time (image build steps
don't have Modal API credentials available to them) -- the `ensemble-aware-
admet` volume is mounted read-only at runtime instead, and each call symlinks
just the one checkpoint file its model_name needs, mirroring
scripts/modal_embed_chemeleon.py exactly.

Before running for the first time, data/prepared (and, for the frozen-
embedding comparison already in data/errors/data/finetune_errors) must
already be on the ml-embedding-benchmark Volume -- confirmed present. Pull
results back locally with:

    modal volume get ml-embedding-benchmark /finetune_errors data/finetune_errors
    modal volume get ml-embedding-benchmark /finetune_checkpoints data/finetune_checkpoints
"""

import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VOLUME_NAME = "ml-embedding-benchmark"
CHECKPOINT_VOLUME_NAME = "ensemble-aware-admet"
MOUNT = pathlib.Path("/data")
CHECKPOINT_MOUNT = pathlib.Path("/ensemble-aware-admet")

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

# model_name -> (remote path on the ensemble-aware-admet volume, local
# filename expected by model_wrappers/chemeleon/finetune.py's _VARIANTS).
# Kept in sync by hand, same convention scripts/modal_embed_chemeleon.py's
# CHECKPOINTS dict already uses.
CHECKPOINTS = {
    "chemeleon_original": ("checkpoints/chemeleon/chemeleon_mp.pt", "chemeleon_mp.pt"),
    "chemeleon_pubchem_3d_mean": ("checkpoints/test-pubchem-mean-3d-only/best.ckpt", "chemeleon_pubchem_3d_mean.ckpt"),
    "chemeleon_pubchem_3d_boltzmann": ("checkpoints/test-pubchem-boltzmann-3d-only/best.ckpt", "chemeleon_pubchem_3d_boltzmann.ckpt"),
    "chemeleon_pubchem_both_mean": ("checkpoints/test-pubchem-3d-and-mordred-mean/best.ckpt", "chemeleon_pubchem_both_mean.ckpt"),
    "chemeleon_pubchem_both_boltzmann": ("checkpoints/test-pubchem-3d-and-mordred-boltzmann/best.ckpt", "chemeleon_pubchem_both_boltzmann.ckpt"),
    "chemeleon_chembl3d_both_boltzmann": ("checkpoints/test-chembl3d-boltzmann-both/best.ckpt", "chemeleon_chembl3d_both_boltzmann.ckpt"),
    "chemeleon_geomdrugs_both_boltzmann": ("checkpoints/test-geom-drugs-boltzmann-both/best.ckpt", "chemeleon_geomdrugs_both_boltzmann.ckpt"),
}

app = modal.App("ml-embedding-benchmark-chemeleon-finetune")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/chemeleon"
    return (
        modal.Image.debian_slim(python_version="3.11")  # matches chemeleon's own .python-version
        .pip_install(
            "joblib", "pandas", "numpy", "rdkit", "pyyaml",
            # Pinned for the same reason as init.sh: chemprop's own torch>=2.1
            # dependency is unpinned and would otherwise resolve a wheel
            # needing a newer CUDA driver than some GPU nodes have.
            "torch==2.5.1", "chemprop==2.2.4",
        )
        .add_local_dir(
            PROJECT_ROOT / "model_wrappers" / "chemeleon",
            remote_path=wrapper_dir,
            ignore=["venv", "venv/**", "checkpoints", "checkpoints/**"],
        )
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
    )


def _run_finetune_chemeleon(
    model_name: str,
    dataset_name: str,
    epochs: int,
    batch_size: int = 64,
    patience: int = 30,
    run_tag: str = "",
    warmup_epochs: int = None,
    force: bool = False,
) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/chemeleon"

    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "finetune_errors").mkdir(parents=True, exist_ok=True)
    (MOUNT / "finetune_checkpoints").mkdir(parents=True, exist_ok=True)
    log_dir = MOUNT / "finetune_logs" / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)
    # Modal reuses a warm container across multiple sequential inputs -- the
    # symlink from the first input on a given container already exists by
    # the time a later input runs there, so this must be idempotent.
    for src, dst in [
        (MOUNT / "prepared", f"{project_root}/data/prepared"),
        (MOUNT / "finetune_errors", f"{project_root}/data/finetune_errors"),
        (MOUNT / "finetune_checkpoints", f"{project_root}/data/finetune_checkpoints"),
    ]:
        if not os.path.lexists(dst):
            os.symlink(str(src), dst, target_is_directory=True)

    remote_path, local_name = CHECKPOINTS[model_name]
    checkpoint_path = f"{wrapper_dir}/checkpoints/{local_name}"
    os.makedirs(f"{wrapper_dir}/checkpoints", exist_ok=True)
    if not os.path.lexists(checkpoint_path):
        os.symlink(str(CHECKPOINT_MOUNT / remote_path), checkpoint_path)

    env = dict(os.environ)
    env["PYTHONPATH"] = project_root

    cmd = [
        "python", "-u", f"{wrapper_dir}/finetune.py",
        "--dataset", dataset_name,
        "--model-name", model_name,
        "--checkpoint-path", checkpoint_path,
        "--config-dir", f"{project_root}/config/dataset",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--patience", str(patience),
    ]
    if run_tag:
        cmd += ["--run-tag", run_tag]
    if warmup_epochs is not None:
        cmd += ["--warmup-epochs", str(warmup_epochs)]
    if force:
        cmd.append("--override")

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    # Written to the Volume rather than relied on via `print()` -- a
    # Lightning-heavy subprocess can flood/truncate Modal's own captured log
    # before the real traceback ever shows up there (same issue
    # scripts/modal_finetune_unimol.py already works around).
    log_name = f"{model_name}_{run_tag}" if run_tag else model_name
    log_path = log_dir / f"{log_name}.log"
    log_path.write_text(f"=== stdout ===\n{result.stdout}\n=== stderr ===\n{result.stderr}\n")
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(
            f"finetune.py failed for dataset={dataset_name} model={model_name} run_tag={run_tag!r} (exit {result.returncode}) "
            f"-- see /finetune_logs/{dataset_name}/{log_name}.log on the Volume"
        )
    return f"ok  {dataset_name}  {model_name}  {run_tag}"


IMAGE = build_image()

finetune_chemeleon = app.function(
    name="finetune_chemeleon",
    image=IMAGE,
    volumes={str(MOUNT): volume, str(CHECKPOINT_MOUNT): checkpoint_volume},
    cpu=4.0,
    memory=8192,
    timeout=60 * 60 * 3,
)(_run_finetune_chemeleon)


@app.local_entrypoint()
def main(
    models: str = ",".join(CHECKPOINTS),
    datasets: str = ",".join(DATASETS),
    epochs: int = 1000,
    batch_size: int = 64,
    patience: int = 30,
    run_tag: str = "",
    modal_gpu: str = "",
    warmup_epochs: int = 0,
    force: bool = False,
):
    """--models: comma-separated model_name values (default: all 7).
    --datasets: comma-separated dataset names (default: all 17).
    Spawns one Modal container per (model, dataset) pair -- all 7 x 17 = 119 by default.
    --epochs: max epochs, early-stopped on val_loss (default 1000).
    --batch-size / --patience: see model_wrappers/chemeleon/finetune.py defaults.
    --run-tag: optional label for a hyperparameter-sweep run, so it's saved
    alongside the default config's result under
    data/finetune_errors/<dataset>/<model>_<run_tag>__finetuned.joblib instead
    of overwriting it.
    --modal-gpu: override GPU type (default: none, CPU-only).
    --warmup-epochs: override DEFAULT_HPARAMS' warmup_epochs (0 = use the default).
    --force: recompute and overwrite even if a cached result already exists."""
    selected_models = [m.strip() for m in models.split(",") if m.strip()]
    unknown_models = set(selected_models) - set(CHECKPOINTS)
    if unknown_models:
        raise ValueError(f"Unknown model(s): {sorted(unknown_models)}. Known: {sorted(CHECKPOINTS)}")

    selected_datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown_datasets = set(selected_datasets) - set(DATASETS)
    if unknown_datasets:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown_datasets)}. Known: {DATASETS}")

    fn = finetune_chemeleon
    if modal_gpu:
        fn = fn.with_options(gpu=None if modal_gpu.lower() == "none" else modal_gpu)

    calls = {
        (m, d): fn.spawn(
            m, d, epochs, batch_size=batch_size, patience=patience, run_tag=run_tag,
            warmup_epochs=warmup_epochs or None, force=force,
        )
        for m in selected_models
        for d in selected_datasets
    }
    for (m, d), call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {m}  {d}: {e}")
