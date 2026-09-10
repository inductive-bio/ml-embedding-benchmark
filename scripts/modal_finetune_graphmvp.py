"""Fine-tune GraphMVP end-to-end on Modal, one CPU container per dataset --
see docs/finetuning_plan.md's Tier 2 rollout.

    modal run scripts/modal_finetune_graphmvp.py --datasets reg_openadmet_pxr_pEC50
    modal run scripts/modal_finetune_graphmvp.py

Mirrors scripts/modal_finetune_molbert.py's structure exactly (see that
file's docstring for the full reasoning): a separate Python 3.7 interpreter
lives side by side with the image's own (Modal-compatible) Python 3.11,
copied straight out of the official python:3.7-bullseye image rather than
compiled from source, and finetune.py runs under it via subprocess.

Also mirrors the split-cache pre-population step: `data/prepared/*.joblib`
is pickled with this repo's pandas 2.3.3, but graphmvp's Python 3.7 venv
can only install pandas up to 1.3.5 (the last cp37 wheel), which can't
unpickle a 2.x DataFrame's internal BlockManager format at all (confirmed
live against model_wrappers/molbert -- the identical problem). The OUTER
(Modal-required >=3.10) driver calls load_finetune_splits() itself before
invoking the subprocess, so the legacy interpreter only ever reads the
resulting CSV cache. See src/common/finetune_data.py's module docstring.

Unlike molbert, graphmvp's own torch pin is CPU-only by design (+cpu wheel,
matching +cpu torch_scatter/torch_sparse/torch_cluster) -- no --gpus flag
here, no GPU requested on the Modal function.

Only `GraphMVP_AM`/`GraphMVP_CP` are fine-tunable (see finetune.py's module
docstring for why the other 8 embedding variants aren't).

Before running for the first time, data/prepared must already be on the
Volume. Pull results back locally with:

    modal volume get ml-embedding-benchmark /finetune_errors data/finetune_errors
    modal volume get ml-embedding-benchmark /finetune_checkpoints data/finetune_checkpoints
"""

import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPHMVP_DIR = PROJECT_ROOT / "model_wrappers" / "graphmvp"
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
VARIANTS = ["GraphMVP_AM", "GraphMVP_CP"]

PY37_VENV = "/opt/graphmvp_venv/bin/python"

app = modal.App("ml-embedding-benchmark-graphmvp-finetune")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    image = modal.Image.from_dockerfile(
        GRAPHMVP_DIR / "Dockerfile.modal",
        context_dir=GRAPHMVP_DIR,
        ignore=["venv", "venv/**", "__pycache__", "__pycache__/**"],
    )
    return (
        image
        # OUTER >=3.10 interpreter -- only used to pre-populate
        # src/common/finetune_data.py's CSV split cache, see module
        # docstring. torch needed too: unpickling data/prepared/*.joblib
        # resolves to src.common.types.Dataset, which imports torch at
        # module level.
        .pip_install("joblib", "pandas", "numpy", "pyyaml", "torch")
        .add_local_file(GRAPHMVP_DIR / "finetune.py", remote_path="/project/model_wrappers/graphmvp/finetune.py")
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
    )


def _run_finetune_graphmvp(
    dataset_name: str,
    variant: str,
    epochs: int,
    run_tag: str = "",
    force: bool = False,
) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/graphmvp"
    checkpoint_path = f"/opt/graphmvp_weights/regression/{variant}/pretraining_model.pth"

    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "finetune_errors").mkdir(parents=True, exist_ok=True)
    (MOUNT / "finetune_checkpoints").mkdir(parents=True, exist_ok=True)
    (MOUNT / "finetune_splits").mkdir(parents=True, exist_ok=True)
    log_dir = MOUNT / "finetune_logs" / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)
    # Idempotent -- see scripts/modal_finetune_unimol.py et al., Modal reuses
    # a warm container across multiple sequential inputs.
    for src, dst in [
        (MOUNT / "prepared", f"{project_root}/data/prepared"),
        (MOUNT / "finetune_errors", f"{project_root}/data/finetune_errors"),
        (MOUNT / "finetune_checkpoints", f"{project_root}/data/finetune_checkpoints"),
        (MOUNT / "finetune_splits", f"{project_root}/data/finetune_splits"),
    ]:
        if not os.path.lexists(dst):
            os.symlink(str(src), dst, target_is_directory=True)

    # Pre-populate the split cache using THIS (outer, modern-pandas)
    # interpreter -- see module docstring.
    import sys
    sys.path.insert(0, project_root)
    import yaml
    from src.common.finetune_data import load_finetune_splits

    with open(f"{project_root}/config/dataset/{dataset_name}.yaml") as f:
        target_column = yaml.safe_load(f)["source"]["target_column"]
    load_finetune_splits(
        dataset_name=dataset_name,
        target_column=target_column,
        prepared_dir=f"{project_root}/data/prepared",
        cache_dir=f"{project_root}/data/finetune_splits",
    )

    env = dict(os.environ)
    # Modal injects its own (much newer) client deps onto PYTHONPATH for the
    # outer >=3.10 interpreter -- python3.7 inheriting that same PYTHONPATH
    # picks up a typing_extensions build using syntax invalid pre-3.8 and
    # crashes on `import torch` (confirmed live against model_wrappers/
    # molbert). finetune.py finds src/ and graphmvp_repo/ itself via
    # sys.path.insert(), so it needs no PYTHONPATH at all.
    env.pop("PYTHONPATH", None)

    cmd = [
        PY37_VENV, "-u", f"{wrapper_dir}/finetune.py",
        "--dataset", dataset_name,
        "--variant", variant,
        "--config-dir", f"{project_root}/config/dataset",
        "--checkpoint-path", checkpoint_path,
        "--graphmvp-repo-dir", "/opt/graphmvp_repo",
        "--epochs", str(epochs),
    ]
    if run_tag:
        cmd += ["--run-tag", run_tag]
    if force:
        cmd.append("--override")

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    log_name = f"graphmvp_{variant}_{run_tag}" if run_tag else f"graphmvp_{variant}"
    log_path = log_dir / f"{log_name}.log"
    log_path.write_text(f"=== stdout ===\n{result.stdout}\n=== stderr ===\n{result.stderr}\n")
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(
            f"finetune.py failed for dataset={dataset_name} variant={variant} run_tag={run_tag!r} "
            f"(exit {result.returncode}) -- see /finetune_logs/{dataset_name}/{log_name}.log on the Volume"
        )
    return f"ok  {dataset_name}  {variant}  {run_tag}"


IMAGE = build_image()

finetune_graphmvp = app.function(
    name="finetune_graphmvp",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    cpu=4.0,
    timeout=60 * 60 * 3,
)(_run_finetune_graphmvp)


@app.local_entrypoint()
def main(
    datasets: str = ",".join(DATASETS),
    variants: str = "GraphMVP_CP",
    epochs: int = 100,
    run_tag: str = "",
    force: bool = False,
):
    """--datasets: comma-separated dataset names (default: all 17).
    --variants: comma-separated GraphMVP variants (default: 'GraphMVP_CP' only --
    pass 'GraphMVP_AM,GraphMVP_CP' for both; these are the only two fine-tunable
    variants, see model_wrappers/graphmvp/finetune.py's module docstring).
    --epochs: default 100, config.py's own --epochs default (EarlyStopping with
    patience=5 on val MAE usually stops well before this).
    --run-tag: optional label for a hyperparameter-sweep run.
    --force: recompute and overwrite even if a cached result already exists."""
    selected_datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected_datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    selected_variants = [v.strip() for v in variants.split(",") if v.strip()]
    unknown_variants = set(selected_variants) - set(VARIANTS)
    if unknown_variants:
        raise ValueError(f"Unknown variant(s): {sorted(unknown_variants)}. Known: {VARIANTS}")

    calls = {
        (d, v): finetune_graphmvp.spawn(d, v, epochs, run_tag=run_tag, force=force)
        for d in selected_datasets
        for v in selected_variants
    }
    for (d, v), call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {d}  {v}: {e}")
