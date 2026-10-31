"""Fine-tune SimSon's pretrained Xtransformer encoder end-to-end on Modal,
one T4 container per dataset.

    modal run scripts/modal_finetune_simson.py --datasets reg_openadmet_pxr_pEC50
    modal run scripts/modal_finetune_simson.py

Unlike molbert (Python 3.7, needs a Dockerfile.modal with a second
side-by-side interpreter -- see model_wrappers/molbert/Dockerfile.modal),
simson's own pin (Python 3.9, torch==2.3.1) is close enough to Modal's own
>=3.10 floor that just bumping the image's python_version works directly,
same pattern as scripts/modal_finetune_unimol.py. Also unlike molbert:
simson's local venv already has pandas==2.3.3 (matching this repo's own
pandas exactly, confirmed live) -- data/prepared/*.joblib unpickles fine, no
CSV split-cache pre-population workaround needed.

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

app = modal.App("ml-embedding-benchmark-simson-finetune")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/simson"
    return (
        modal.Image.debian_slim(python_version="3.11")  # bumped from simson's local 3.9 pin, same reasoning as modal_finetune_unimol.py
        .apt_install("git")
        .pip_install(
            "joblib", "tqdm", "pandas", "numpy<2.0.0,>=1.26.4", "rdkit",
            "setuptools<81", "scipy", "scikit-learn", "pyyaml",
        )
        # Exact versions per model_wrappers/simson/requirements.txt, except
        # numpy (that file pins 1.22.4, predating cp311 wheels entirely --
        # nothing in this wrapper is numerically sensitive to that specific
        # patch version) and torchaudio/torchvision (unused by finetune.py).
        .pip_install(
            "torch==2.3.1", "tokenizers==0.19.1", "transformers==4.42.3", "x-transformers==1.31.6",
        )
        .add_local_dir(
            PROJECT_ROOT / "model_wrappers" / "simson", remote_path=wrapper_dir, copy=True,
            ignore=["venv", "venv/**", "__pycache__", "__pycache__/**"],
        )
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
    )


def _run_finetune_simson(
    dataset_name: str,
    epochs: int,
    batch_size: int = 128,
    run_tag: str = "",
    force: bool = False,
) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/simson"

    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "finetune_errors").mkdir(parents=True, exist_ok=True)
    (MOUNT / "finetune_checkpoints").mkdir(parents=True, exist_ok=True)
    log_dir = MOUNT / "finetune_logs" / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)
    # Idempotent -- Modal reuses a warm container across multiple sequential
    # inputs; an unconditional os.symlink crashes on the second call in a
    # container (confirmed live against several other modal_finetune_*.py).
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
        "--batch-size", str(batch_size),
    ]
    if run_tag:
        cmd += ["--run-tag", run_tag]
    if force:
        cmd.append("--override")

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    log_name = f"simson_{run_tag}" if run_tag else "simson"
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

finetune_simson = app.function(
    name="finetune_simson",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    # T4 (16GB) CUDA-OOMs on config.py's own batch_size=128 default at
    # max_len=512 (confirmed live) -- A10G (24GB) rather than shrinking the
    # batch size away from the documented default.
    gpu="A10G",
    cpu=4.0,
    # 3h wasn't enough for reg_openadmet_expansionrx_LogD (confirmed timed
    # out live) -- finetune.py only torch.save()s a checkpoint *after* the
    # full training loop (early-stopped or all `epochs`) completes, so a
    # timeout leaves no partial artifact and no log to diagnose from either
    # (the Modal-side log write also only happens after subprocess.run()
    # returns). Bumped rather than reducing epochs/patience away from
    # config.py's own documented defaults.
    timeout=60 * 60 * 6,
)(_run_finetune_simson)


@app.local_entrypoint()
def main(
    datasets: str = ",".join(DATASETS),
    epochs: int = 500,
    batch_size: int = 128,
    run_tag: str = "",
    modal_gpu: str = "",
    force: bool = False,
):
    """--datasets: comma-separated dataset names (default: all 17).
    --epochs: default 500, config.py's own --epochs default (EarlyStopping
    with patience=10 on val RMSE usually stops well before this).
    --batch-size: default 128, config.py's own --batch_size default (needs
    the A10G this script now defaults to -- T4 CUDA-OOMs at this batch size).
    --run-tag: optional label for a hyperparameter-sweep run.
    --modal-gpu: override GPU type (default: A10G). Pass "none" for CPU-only.
    --force: recompute and overwrite even if a cached result already exists."""
    selected_datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected_datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    fn = finetune_simson
    if modal_gpu:
        fn = fn.with_options(gpu=None if modal_gpu.lower() == "none" else modal_gpu)

    calls = {d: fn.spawn(d, epochs, batch_size=batch_size, run_tag=run_tag, force=force) for d in selected_datasets}
    for d, call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {d}: {e}")
