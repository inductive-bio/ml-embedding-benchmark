"""Fine-tune MolBERT end-to-end on Modal, one T4 container per dataset --
see docs/finetuning_plan.md's Tier 2 rollout.

    modal run scripts/modal_finetune_molbert.py --datasets reg_openadmet_pxr_pEC50
    modal run scripts/modal_finetune_molbert.py

Unlike every other scripts/modal_finetune_<name>.py, this can't just bump
its wrapper's own Python version for the Modal image the way
scripts/modal_embed.py does for fingerprints/mol2vec/clamp/coati/gem/
huggingmolecules -- molbert_repo/setup.py pins torch==1.4.0/
pytorch-lightning==0.8.4/transformers==3.5.1, none of which ship a wheel for
Python >=3.10 (which Modal's own client library requires). See
model_wrappers/molbert/Dockerfile.modal for how this is solved: a separate
Python 3.7 interpreter lives side by side with the image's own (Modal-
compatible) Python 3.11, copied straight out of the official
python:3.7-bullseye image rather than compiled from source (~72s vs. ~28
minutes for an identical result, confirmed empirically) -- and
finetune.py runs under that interpreter via subprocess, same pattern as
every other modal_finetune_<name>.py's subprocess call, just pointed at a
different interpreter than the container's own.

Also unlike every other modal_finetune_<name>.py: before invoking the
subprocess, _run_finetune_molbert() calls load_finetune_splits() itself
using the OUTER (Modal-required >=3.10) interpreter, to pre-populate
src/common/finetune_data.py's CSV split cache -- the inner Python 3.7 venv's
pandas (1.3.5, the last cp37 wheel) can't unpickle data/prepared/*.joblib at
all (it was written by this repo's own pandas 2.3.3; confirmed live:
`AttributeError: Can't get attribute '_unpickle_block'`), so the split must
be derived under modern pandas first. See that module's docstring.

Before running for the first time, data/prepared must already be on the
Volume. Pull results back locally with:

    modal volume get ml-embedding-benchmark /finetune_errors data/finetune_errors
    modal volume get ml-embedding-benchmark /finetune_checkpoints data/finetune_checkpoints
"""

import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MOLBERT_DIR = PROJECT_ROOT / "model_wrappers" / "molbert"
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

PY37_VENV = "/opt/molbert_venv/bin/python"
PRETRAINED_MODEL_PATH = "/opt/molbert_weights/molbert_100epochs/checkpoints/last.ckpt"

app = modal.App("ml-embedding-benchmark-molbert-finetune")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    image = modal.Image.from_dockerfile(
        MOLBERT_DIR / "Dockerfile.modal",
        context_dir=MOLBERT_DIR,
        # weights.zip is the same ~967MB weights/ already extracted from,
        # and venv/__pycache__ are >1GB of irrelevant local-only artifacts --
        # excluded from the build context entirely rather than just from the
        # final image, so they're not even scanned/uploaded for the build.
        ignore=["venv", "venv/**", "weights.zip", "__pycache__", "__pycache__/**"],
    )
    return (
        image
        # OUTER >=3.10 interpreter (Modal's requirement) -- used only to
        # pre-populate the CSV split cache before the legacy subprocess (see
        # module docstring). torch is needed too: unpickling
        # data/prepared/*.joblib resolves to src.common.types.Dataset, which
        # imports torch at module level.
        .pip_install("joblib", "pandas", "numpy", "pyyaml", "torch")
        .add_local_file(MOLBERT_DIR / "finetune.py", remote_path="/project/model_wrappers/molbert/finetune.py")
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
    )


def _run_finetune_molbert(
    dataset_name: str,
    max_epochs: int,
    gpus: int = 1,
    run_tag: str = "",
    force: bool = False,
) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/molbert"

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

    # Pre-populate the CSV split cache with this outer, modern-pandas
    # interpreter -- the legacy subprocess's pandas 1.3.5 can't unpickle
    # data/prepared/*.joblib at all (see module docstring). Once the cache
    # exists, finetune.py's load_finetune_splits() just reads the CSVs.
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
    # Modal injects its own newer client deps onto PYTHONPATH for the outer
    # >=3.10 interpreter -- python3.7 inheriting that PYTHONPATH picks up a
    # typing_extensions build using positional-only-parameter syntax invalid
    # pre-3.8 and crashes on `import torch` (confirmed live). finetune.py
    # finds src/ and molbert_repo/ itself via sys.path.insert(), so no
    # PYTHONPATH is needed.
    env.pop("PYTHONPATH", None)

    cmd = [
        PY37_VENV, "-u", f"{wrapper_dir}/finetune.py",
        "--dataset", dataset_name,
        "--config-dir", f"{project_root}/config/dataset",
        "--pretrained-model-path", PRETRAINED_MODEL_PATH,
        "--max-epochs", str(max_epochs),
        "--gpus", str(gpus),
    ]
    if run_tag:
        cmd += ["--run-tag", run_tag]
    if force:
        cmd.append("--override")

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    log_name = f"molbert_{run_tag}" if run_tag else "molbert"
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

finetune_molbert = app.function(
    name="finetune_molbert",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    gpu="T4",
    cpu=4.0,
    timeout=60 * 60 * 3,
)(_run_finetune_molbert)


@app.local_entrypoint()
def main(
    datasets: str = ",".join(DATASETS),
    max_epochs: int = 20,
    gpus: int = 1,
    run_tag: str = "",
    modal_gpu: str = "",
    force: bool = False,
):
    """--datasets: comma-separated dataset names (default: all 17).
    --max-epochs: default 20, apps/args.py's own --max_epochs default
    (EarlyStopping with patience=5 on valid_loss usually stops well before this).
    --gpus: GPUs per container for pytorch_lightning.Trainer (default 1).
    --run-tag: optional label for a hyperparameter-sweep run.
    --modal-gpu: override GPU type (default: T4). Pass "none" for CPU-only
    (also pass --gpus 0 in that case).
    --force: recompute and overwrite even if a cached result already exists."""
    selected_datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected_datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    fn = finetune_molbert
    if modal_gpu:
        fn = fn.with_options(gpu=None if modal_gpu.lower() == "none" else modal_gpu)

    calls = {
        d: fn.spawn(d, max_epochs, gpus=gpus, run_tag=run_tag, force=force)
        for d in selected_datasets
    }
    for d, call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {d}: {e}")
