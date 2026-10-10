"""Fine-tune Atom-JEPA end-to-end on Modal, one container per dataset.

    modal run scripts/modal_finetune_atom_jepa.py --datasets reg_openadmet_pxr_pEC50
    modal run scripts/modal_finetune_atom_jepa.py --modal-gpu A100
    modal run scripts/modal_finetune_atom_jepa.py --modal-gpu T4 --fp32

Defaults to an A10G so the encoder runs in bf16 with cuEquivariance kernels.
On T4/V100 pass --fp32 (loads with `precision="fp32", cuequivariance=False`).

data/prepared must already be on the Volume (see modal_embed_atom_jepa.py).
Pull results back locally with:

    modal volume get ml-embedding-benchmark /finetune_errors data/finetune_errors
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

app = modal.App("ml-embedding-benchmark-atom-jepa-finetune")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/atom_jepa"
    return (
        modal.Image.debian_slim(python_version="3.12")  # matches atom_jepa's .python-version
        .apt_install("libxrender1", "libxext6", "libsm6")
        .pip_install("torch")  # atom-jepa requires PyTorch to be installed first
        .pip_install(
            "joblib", "tqdm", "pandas", "numpy", "scipy", "scikit-learn",
            "pyyaml", "setuptools<81",
        )
        .add_local_dir(PROJECT_ROOT / "model_wrappers" / "atom_jepa", remote_path=wrapper_dir, copy=True)
        .run_commands(f"cd {wrapper_dir} && ATOM_JEPA_EXTRAS=rdkit,cu12 INSTALL_DEP=true bash -e init.sh")
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config" / "dataset", remote_path="/project/config/dataset")
    )


def _run_finetune_atom_jepa(
    dataset_name: str,
    epochs: int = 30,
    encoder_lr: float = 1e-5,
    fp32: bool = False,
    run_tag: str = "",
    force: bool = False,
) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/atom_jepa"

    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "finetune_errors").mkdir(parents=True, exist_ok=True)
    log_dir = MOUNT / "finetune_logs" / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)
    # Modal reuses a warm container across inputs, so symlinks must be idempotent.
    for src, dst in [
        (MOUNT / "prepared", f"{project_root}/data/prepared"),
        (MOUNT / "finetune_errors", f"{project_root}/data/finetune_errors"),
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
        "--encoder-lr", str(encoder_lr),
    ]
    if fp32:
        cmd.append("--fp32")
    if run_tag:
        cmd += ["--run-tag", run_tag]
    if force:
        cmd.append("--override")

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    # Written to the Volume because Modal's captured log can truncate before
    # the real traceback shows up.
    log_name = f"atom_jepa_{run_tag}" if run_tag else "atom_jepa"
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

finetune_atom_jepa = app.function(
    name="finetune_atom_jepa",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    gpu="A10G",
    cpu=4.0,
    timeout=60 * 60 * 3,
)(_run_finetune_atom_jepa)


@app.local_entrypoint()
def main(
    datasets: str = ",".join(DATASETS),
    epochs: int = 30,
    encoder_lr: float = 1e-5,
    fp32: bool = False,
    run_tag: str = "",
    modal_gpu: str = "",
    force: bool = False,
):
    """--datasets: comma-separated dataset names (default: all 17).
    --epochs / --encoder-lr: forwarded to finetune.py.
    --fp32: fp32 without cuEquivariance, needed on T4/V100.
    --run-tag: label for a sweep run, saved alongside the default result.
    --modal-gpu: override GPU type (default: A10G). Pass "none" for CPU-only.
    --force: recompute and overwrite an existing result."""
    selected = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    fn = finetune_atom_jepa
    if modal_gpu:
        fn = fn.with_options(gpu=None if modal_gpu.lower() == "none" else modal_gpu)

    calls = {
        d: fn.spawn(d, epochs, encoder_lr=encoder_lr, fp32=fp32, run_tag=run_tag, force=force)
        for d in selected
    }
    for d, call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {d}: {e}")
