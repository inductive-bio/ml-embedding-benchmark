"""Fine-tune GEM (GeoGNN) end-to-end on Modal, one container per dataset --
see docs/finetuning_plan.md's Tier 1 pilot step.

    modal run scripts/modal_finetune_gem.py --datasets reg_openadmet_pxr_pEC50 --epochs 5
    modal run scripts/modal_finetune_gem.py

Mirrors scripts/modal_finetune_unimol.py's structure. Unlike Uni-Mol, GEM's
paddlepaddle install is CPU-only (requirements.txt pins "paddlepaddle>=2.0.0",
not the -gpu variant -- confirmed already true for GEM's embedding image in
scripts/modal_embed.py's GPU_FOR dict, which excludes "gem"), so no GPU is
requested here; more CPU cores are requested instead. Also mirrors
modal_embed.py's Python-version choice for GEM specifically: local venv is
3.8, but Modal's client library requires >=3.10, so the *Modal* image uses
3.11 (confirmed working live for GEM's embedding image already).

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

app = modal.App("ml-embedding-benchmark-gem-finetune")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/gem"
    return (
        modal.Image.debian_slim(python_version="3.11")  # GEM's local pin (3.8) bumped for Modal's client lib, same as modal_embed.py does
        .apt_install("git")
        .pip_install(
            "hydra-core", "hydra-joblib-launcher", "torch", "joblib", "tqdm",
            "pandas", "numpy<2.0.0,>=1.26.4", "rdkit", "setuptools<81",
            "scipy", "scikit-learn", "pyyaml",
        )
        .add_local_dir(
            PROJECT_ROOT / "model_wrappers" / "gem", remote_path=wrapper_dir, copy=True,
            ignore=["venv", "venv/**"],
        )
        .run_commands(f"cd {wrapper_dir} && INSTALL_DEP=true bash -e init.sh")
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
    )


def _run_finetune_gem(
    dataset_name: str,
    epochs: int,
    encoder_lr: float = 1e-3,
    head_lr: float = 1e-3,
    run_tag: str = "",
    force: bool = False,
) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/gem"

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

    # Deliberately NOT setting PYTHONPATH to project_root here: this repo's
    # own src/ (a regular package, has __init__.py) would then shadow GEM's
    # vendored src/ (a namespace package, no __init__.py) for GEM's own
    # internal `from src.model import ...`-style imports -- a regular
    # package always wins over a namespace portion regardless of sys.path
    # order, confirmed live (`ModuleNotFoundError: No module named
    # 'src.featurizer'`). finetune.py adds GEM's own directories to sys.path
    # itself and loads our shared finetune_data/finetune_results modules via
    # direct file-path importlib instead, so this isn't needed.
    env = dict(os.environ)

    cmd = [
        "python", "-u", f"{wrapper_dir}/finetune.py",
        "--dataset", dataset_name,
        "--config-dir", f"{project_root}/config/dataset",
        "--epochs", str(epochs),
        "--encoder-lr", str(encoder_lr),
        "--head-lr", str(head_lr),
    ]
    if run_tag:
        cmd += ["--run-tag", run_tag]
    if force:
        cmd.append("--override")

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    # Written to the Volume rather than relied on via `print()` -- Modal's
    # captured log can get flooded/truncated by a verbose subprocess before
    # the real traceback ever shows up there (hit this for real with Uni-Mol).
    log_name = f"GEM_{run_tag}" if run_tag else "GEM"
    log_path = log_dir / f"{log_name}.log"
    log_path.write_text(f"=== stdout ===\n{result.stdout}\n=== stderr ===\n{result.stderr}\n")
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(
            f"finetune.py failed for dataset={dataset_name} run_tag={run_tag!r} (exit {result.returncode}) "
            f"-- see /finetune_logs/{dataset_name}/{log_name}.log on the Volume"
        )
    return f"ok  {dataset_name}  GEM  {run_tag}"


IMAGE = build_image()

finetune_gem = app.function(
    name="finetune_gem",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    cpu=8.0,  # CPU-only paddlepaddle wheel -- no GPU benefit, see module docstring
    timeout=60 * 60 * 5,
)(_run_finetune_gem)


@app.local_entrypoint()
def main(
    datasets: str = ",".join(DATASETS),
    epochs: int = 100,
    encoder_lr: float = 1e-3,
    head_lr: float = 1e-3,
    run_tag: str = "",
    force: bool = False,
):
    """--datasets: comma-separated dataset names (default: all 17).
    --epochs: default 100, GEM's own scripts/finetune_regr.sh default -- a
    long CPU-only run per dataset; override for a cheaper smoke test.
    --encoder-lr/--head-lr: default 1e-3/1e-3, GEM's own first hyperparameter
    combo in scripts/finetune_regr.sh's sweep.
    --run-tag: optional label for a hyperparameter-sweep run.
    --force: recompute and overwrite even if a cached result already exists."""
    selected_datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected_datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    calls = {
        d: finetune_gem.spawn(d, epochs, encoder_lr=encoder_lr, head_lr=head_lr, run_tag=run_tag, force=force)
        for d in selected_datasets
    }
    for d, call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {d}: {e}")
