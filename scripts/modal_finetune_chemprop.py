"""Train a chemprop MPNN end-to-end (2D and/or 3D features, no pretraining)
on Modal, one container per (dataset, features) -- see
docs/e2e_chemprop_chemeleon_plan.md.

    modal run scripts/modal_finetune_chemprop.py --datasets reg_openadmet_polaris_admet_KSOL --features 2d
    modal run scripts/modal_finetune_chemprop.py --features 2d,3d
    modal run scripts/modal_finetune_chemprop.py --modal-gpu T4

Before running for the first time, data/prepared (and, for --features 3d,
data/conformers/<challenge>_mmff94s.joblib) must already be on the Volume --
confirmed present as of this writing. Pull results back locally with:

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
FEATURE_VARIANTS = ["2d", "3d"]

app = modal.App("ml-embedding-benchmark-chemprop-finetune")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/chemprop"
    return (
        modal.Image.debian_slim(python_version="3.11")  # matches chemprop's own .python-version
        .pip_install("joblib", "pandas", "numpy", "rdkit", "scikit-learn", "pyyaml", "tqdm")
        .add_local_dir(PROJECT_ROOT / "model_wrappers" / "chemprop", remote_path=wrapper_dir, copy=True)
        .run_commands(f"cd {wrapper_dir} && INSTALL_DEP=true bash -e init.sh")
        .add_local_file(
            PROJECT_ROOT / "model_wrappers" / "descriptors" / "rdkit_3d.py",
            remote_path="/project/model_wrappers/descriptors/rdkit_3d.py",
        )
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
    )


def _run_finetune_chemprop(
    dataset_name: str,
    features: str,
    epochs: int,
    batch_size: int = 64,
    patience: int = 30,
    run_tag: str = "",
    warmup_epochs: int = None,
    mpnn_hidden_dim: int = None,
    force: bool = False,
) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/chemprop"

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
        (MOUNT / "conformers", f"{project_root}/data/conformers"),
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
        "--features", features,
        "--config-dir", f"{project_root}/config/dataset",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--patience", str(patience),
    ]
    if run_tag:
        cmd += ["--run-tag", run_tag]
    if warmup_epochs is not None:
        cmd += ["--warmup-epochs", str(warmup_epochs)]
    if mpnn_hidden_dim is not None:
        cmd += ["--mpnn-hidden-dim", str(mpnn_hidden_dim)]
    if force:
        cmd.append("--override")

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    # Written to the Volume rather than relied on via `print()` -- a
    # Lightning-heavy subprocess can flood/truncate Modal's own captured log
    # before the real traceback ever shows up there (same issue
    # scripts/modal_finetune_unimol.py already works around).
    model_name = f"chemprop_{features}"
    log_name = f"{model_name}_{run_tag}" if run_tag else model_name
    log_path = log_dir / f"{log_name}.log"
    log_path.write_text(f"=== stdout ===\n{result.stdout}\n=== stderr ===\n{result.stderr}\n")
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(
            f"finetune.py failed for dataset={dataset_name} features={features} run_tag={run_tag!r} (exit {result.returncode}) "
            f"-- see /finetune_logs/{dataset_name}/{log_name}.log on the Volume"
        )
    return f"ok  {dataset_name}  {model_name}  {run_tag}"


IMAGE = build_image()

finetune_chemprop = app.function(
    name="finetune_chemprop",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    cpu=4.0,
    memory=8192,
    timeout=60 * 60 * 3,
)(_run_finetune_chemprop)


@app.local_entrypoint()
def main(
    datasets: str = ",".join(DATASETS),
    features: str = "2d",
    epochs: int = 1000,
    batch_size: int = 64,
    patience: int = 30,
    run_tag: str = "",
    modal_gpu: str = "",
    warmup_epochs: int = 0,
    mpnn_hidden_dim: int = 0,
    force: bool = False,
):
    """--datasets: comma-separated dataset names (default: all 17).
    --features: comma-separated feature variant(s), '2d' and/or '3d' (default: '2d' only).
    --epochs: max epochs, early-stopped on val_loss (default 1000, matching
    ensemble-aware-admet's own train.sh).
    --batch-size / --patience: see model_wrappers/chemprop/finetune.py defaults.
    --run-tag: optional label (e.g. 'epochs300') for a hyperparameter-sweep
    run, so it's saved alongside the default config's result under
    data/finetune_errors/<dataset>/chemprop_<features>_<run_tag>__finetuned.joblib
    instead of overwriting it.
    --modal-gpu: override GPU type (default: none, CPU-only -- chemprop's MPNN
    is small relative to these per-endpoint dataset sizes).
    --warmup-epochs / --mpnn-hidden-dim: override model_wrappers/chemprop/
    finetune.py's DEFAULT_HPARAMS (0 = use the default; for a hyperparameter-
    sweep run, pass --run-tag too so results don't overwrite the default's).
    --force: recompute and overwrite even if a cached result already exists."""
    selected_datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected_datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    selected_features = [f.strip() for f in features.split(",") if f.strip()]
    unknown_features = set(selected_features) - set(FEATURE_VARIANTS)
    if unknown_features:
        raise ValueError(f"Unknown --features value(s): {sorted(unknown_features)}. Known: {FEATURE_VARIANTS}")

    fn = finetune_chemprop
    if modal_gpu:
        fn = fn.with_options(gpu=None if modal_gpu.lower() == "none" else modal_gpu)

    calls = {
        (d, f): fn.spawn(
            d, f, epochs, batch_size=batch_size, patience=patience, run_tag=run_tag,
            warmup_epochs=warmup_epochs or None, mpnn_hidden_dim=mpnn_hidden_dim or None, force=force,
        )
        for d in selected_datasets
        for f in selected_features
    }
    for (d, f), call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {d}  {f}: {e}")
