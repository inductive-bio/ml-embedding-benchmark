"""Fine-tune SELFormer (or SELFormer-Lite) end-to-end on Modal, one container
per (dataset, model variant) -- see docs/finetuning_plan.md's Tier 1 pilot
step. Mirrors scripts/modal_finetune_unimol.py's structure; image mirrors
scripts/modal_embed_selformer.py's (same wrapper, same checkpoint layout).

    modal run scripts/modal_finetune_selformer.py --datasets reg_openadmet_pxr_pEC50
    modal run scripts/modal_finetune_selformer.py --model-names SELFormer,SELFormer-Lite
    modal run scripts/modal_finetune_selformer.py --modal-gpu A10G

Before running for the first time, data/prepared must already be on the
Volume (it is, as of this writing). Pull results back locally with:

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
MODEL_NAMES = ["SELFormer", "SELFormer-Lite"]

app = modal.App("ml-embedding-benchmark-selformer-finetune")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/huggingface"
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git")
        .pip_install(
            "joblib", "tqdm", "pandas", "numpy<2.0.0,>=1.26.4", "rdkit",
            "setuptools<81", "scipy", "scikit-learn", "pyyaml",
        )
        # Same pin as modal_embed_selformer.py -- SELFormer doesn't need
        # transformers>=5.0 (unlike MolFormer), no reason to diverge.
        .pip_install(
            "transformers[torch]==4.46.3", "selfies==2.1.1", "sentencepiece",
            "protobuf", "sentence-transformers",
        )
        .add_local_dir(
            PROJECT_ROOT / "model_wrappers" / "huggingface",
            remote_path=wrapper_dir,
            copy=True,
            ignore=["venv", "venv/**", "venv_molformer", "venv_molformer/**"],
        )
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
    )


def _run_finetune_selformer(
    dataset_name: str,
    model_name: str,
    epochs: int,
    lr: float = 5e-5,
    run_tag: str = "",
    force: bool = False,
) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/huggingface"

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
        "python", "-u", f"{wrapper_dir}/finetune_selformer.py",
        "--dataset", dataset_name,
        "--model-name", model_name,
        "--config-dir", f"{project_root}/config/dataset",
        "--epochs", str(epochs),
        "--lr", str(lr),
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
            f"finetune_selformer.py failed for dataset={dataset_name} model={model_name} run_tag={run_tag!r} "
            f"(exit {result.returncode}) -- see /finetune_logs/{dataset_name}/{log_name}.log on the Volume"
        )
    return f"ok  {dataset_name}  {model_name}  {run_tag}"


IMAGE = build_image()

finetune_selformer = app.function(
    name="finetune_selformer",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    gpu="T4",
    cpu=4.0,
    timeout=60 * 60 * 3,
)(_run_finetune_selformer)


@app.local_entrypoint()
def main(
    datasets: str = ",".join(DATASETS),
    model_names: str = "SELFormer",
    epochs: int = 25,
    lr: float = 5e-5,
    run_tag: str = "",
    modal_gpu: str = "",
    force: bool = False,
):
    """--datasets: comma-separated dataset names (default: all 17).
    --model-names: comma-separated model variants (default: 'SELFormer' only --
    pass 'SELFormer,SELFormer-Lite' for both).
    --epochs: default 25, the SELFormer README's regression recipe.
    --lr: default 5e-5, the SELFormer README's regression recipe.
    --run-tag: optional label for a hyperparameter-sweep run.
    --modal-gpu: override GPU type (default: T4). Pass "none" for CPU-only.
    --force: recompute and overwrite even if a cached result already exists."""
    selected_datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected_datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    selected_models = [m.strip() for m in model_names.split(",") if m.strip()]
    unknown_models = set(selected_models) - set(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"Unknown model name(s): {sorted(unknown_models)}. Known: {MODEL_NAMES}")

    fn = finetune_selformer
    if modal_gpu:
        fn = fn.with_options(gpu=None if modal_gpu.lower() == "none" else modal_gpu)

    calls = {
        (d, m): fn.spawn(d, m, epochs, lr=lr, run_tag=run_tag, force=force)
        for d in selected_datasets
        for m in selected_models
    }
    for (d, m), call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {d}  {m}: {e}")
