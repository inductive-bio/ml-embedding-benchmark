"""Fine-tune MAT/R-MAT/GROVER end-to-end on Modal, one container per
(dataset, model variant).

    modal run scripts/modal_finetune_huggingmolecules.py --datasets reg_openadmet_pxr_pEC50 --model-names mat_masking_20M
    modal run scripts/modal_finetune_huggingmolecules.py --model-names grover_large,rmat_4M

Mirrors scripts/modal_finetune_gem.py's structure; image mirrors
scripts/modal_embed.py's huggingmolecules image (same wrapper, same init.sh,
no extra apt/post-init overrides needed for this one). Python bumped to 3.11
for Modal's own client-library requirement, same as modal_embed.py's
MODAL_PYTHON_VERSION["huggingmolecules"] (local venv is 3.9).

Before running for the first time, data/prepared must already be on the
Volume. Pull results back locally with:

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
# All variants config/experiment/huggingmolecules.yaml sweeps over (used only to
# validate --model-names below); the default `model_names` in main() is
# restricted to the best variant per family -- most pretraining data for MAT
# (20M > 2M > 200k), the larger of the two GROVER checkpoints, and R-MAT's
# only default variant.
MODEL_NAMES = ["mat_masking_200k", "mat_masking_2M", "mat_masking_20M", "grover_base", "grover_large", "rmat_4M"]
BEST_MODEL_NAMES = ["mat_masking_20M", "grover_large", "rmat_4M"]

app = modal.App("ml-embedding-benchmark-huggingmolecules-finetune")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/huggingmolecules"
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install(
            "hydra-core", "hydra-joblib-launcher", "torch", "joblib", "tqdm",
            "pandas", "numpy<2.0.0,>=1.26.4", "rdkit", "setuptools<81",
            "scipy", "scikit-learn", "pyyaml",
        )
        .add_local_dir(
            PROJECT_ROOT / "model_wrappers" / "huggingmolecules", remote_path=wrapper_dir, copy=True,
            ignore=["venv", "venv/**"],
        )
        .run_commands(f"cd {wrapper_dir} && INSTALL_DEP=true bash -e init.sh")
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
    )


def _run_finetune_huggingmolecules(
    dataset_name: str,
    model_name: str,
    epochs: int,
    run_tag: str = "",
    force: bool = False,
) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/huggingmolecules"

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

    env = dict(os.environ)
    env["PYTHONPATH"] = project_root

    cmd = [
        "python", "-u", f"{wrapper_dir}/finetune.py",
        "--dataset", dataset_name,
        "--model-name", model_name,
        "--config-dir", f"{project_root}/config/dataset",
        "--epochs", str(epochs),
    ]
    if run_tag:
        cmd += ["--run-tag", run_tag]
    if force:
        cmd.append("--override")

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    # Written to the Volume rather than relied on via `print()` -- same
    # log-truncation risk flagged in modal_finetune_unimol.py.
    log_name = f"{model_name}_{run_tag}" if run_tag else model_name
    log_path = log_dir / f"{log_name}.log"
    log_path.write_text(f"=== stdout ===\n{result.stdout}\n=== stderr ===\n{result.stderr}\n")
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(
            f"finetune.py failed for dataset={dataset_name} model={model_name} run_tag={run_tag!r} "
            f"(exit {result.returncode}) -- see /finetune_logs/{dataset_name}/{log_name}.log on the Volume"
        )
    return f"ok  {dataset_name}  {model_name}  {run_tag}"


IMAGE = build_image()

finetune_huggingmolecules = app.function(
    name="finetune_huggingmolecules",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    gpu="T4",
    cpu=4.0,
    timeout=60 * 60 * 6,
)(_run_finetune_huggingmolecules)


@app.local_entrypoint()
def main(
    datasets: str = ",".join(DATASETS),
    model_names: str = ",".join(BEST_MODEL_NAMES),
    epochs: int = 100,
    run_tag: str = "",
    modal_gpu: str = "",
    force: bool = False,
):
    """--datasets: comma-separated dataset names (default: all 17).
    --model-names: comma-separated model variants (default: the best variant per
    family -- 'mat_masking_20M,grover_large,rmat_4M'; pass e.g. 'mat_masking_200k'
    to run a smaller/cheaper variant instead). lr/batch_size/weight_decay
    are not exposed here -- finetune.py picks each variant's own documented default
    (see model_wrappers/huggingmolecules/finetune.py's VARIANTS); pass --run-tag and
    invoke finetune.py directly (via a one-off container) for a hyperparameter sweep.
    --epochs: default 100, matching every model family's own
    experiments/configs/models/*.gin num_epochs.
    --run-tag: optional label for a hyperparameter-sweep run.
    --modal-gpu: override GPU type (default: T4). Pass "none" for CPU-only.
    --force: recompute and overwrite even if a cached result already exists."""
    selected_datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected_datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    selected_models = [m.strip() for m in model_names.split(",") if m.strip()]

    fn = finetune_huggingmolecules
    if modal_gpu:
        fn = fn.with_options(gpu=None if modal_gpu.lower() == "none" else modal_gpu)

    calls = {
        (d, m): fn.spawn(d, m, epochs, run_tag=run_tag, force=force)
        for d in selected_datasets
        for m in selected_models
    }
    for (d, m), call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {d}  {m}: {e}")
