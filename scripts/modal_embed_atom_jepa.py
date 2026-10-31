"""Run embed.py on Modal for the `atom_jepa` wrapper, one container per dataset.

    modal run scripts/modal_embed_atom_jepa.py
    modal run scripts/modal_embed_atom_jepa.py --datasets reg_openadmet_pxr_pEC50

Defaults to an A10G (Ampere) so the encoder runs in bf16 with cuEquivariance
kernels (the `cu12` extra). Pre-Ampere GPUs (T4, V100) need
`precision="fp32", cuequivariance=False` per the atom-jepa README, which the
embedding wrapper does not set, so use A10G or newer.

Before running for the first time, upload data/prepared to the Volume if not
already there:

    modal volume put ml-embedding-benchmark data/prepared /prepared

Pull results back locally with:

    modal volume get ml-embedding-benchmark /embedded data/embedded
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

app = modal.App("ml-embedding-benchmark-atom-jepa")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/atom_jepa"
    return (
        modal.Image.debian_slim(python_version="3.12")  # matches atom_jepa's .python-version
        # rdkit.Chem.Draw is imported at load time and needs X11 libs
        # debian_slim doesn't ship (same fix as modal_embed_unimol.py).
        .apt_install("libxrender1", "libxext6", "libsm6")
        # atom-jepa requires PyTorch first; same pin as init.sh.
        .pip_install("torch==2.14.1")
        .pip_install(
            "hydra-core", "hydra-joblib-launcher", "joblib", "tqdm", "pandas",
            "numpy", "setuptools<81",
        )
        .add_local_dir(
            PROJECT_ROOT / "model_wrappers" / "atom_jepa", remote_path=wrapper_dir, copy=True,
            ignore=["venv", "venv/**"],
        )
        .run_commands(f"cd {wrapper_dir} && ATOM_JEPA_EXTRAS=rdkit,cu12 INSTALL_DEP=true bash -e init.sh")
        # repo mount, after run_commands -- Modal forbids run_commands after a
        # non-copy add_local_dir.
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config", remote_path="/project/config")
        .add_local_file(PROJECT_ROOT / "embed.py", remote_path="/project/embed.py")
    )


def _run_embed_atom_jepa(dataset_name: str, force: bool = False) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/atom_jepa"

    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "embedded").mkdir(parents=True, exist_ok=True)
    # Modal reuses a warm container across inputs, so symlinks must be idempotent.
    for src, dst in [
        (MOUNT / "prepared", f"{project_root}/data/prepared"),
        (MOUNT / "embedded", f"{project_root}/data/embedded"),
    ]:
        if not os.path.lexists(dst):
            os.symlink(str(src), dst, target_is_directory=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{project_root}:{wrapper_dir}"

    cmd = [
        "python", "-u", "embed.py",
        "--multirun",
        "+experiment=atom_jepa",
        f"+dataset={dataset_name}",
        f"++cache={'false' if force else 'true'}",
    ]

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"embed.py failed for dataset={dataset_name} (exit {result.returncode})")
    return f"ok  {dataset_name}"


IMAGE = build_image()

embed_atom_jepa = app.function(
    name="embed_atom_jepa",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    gpu="A10G",
    cpu=4.0,  # conformer generation is CPU-bound
    timeout=60 * 60 * 3,
)(_run_embed_atom_jepa)


@app.local_entrypoint()
def main(datasets: str = ",".join(DATASETS), modal_gpu: str = "", modal_cpu: float = 0, force: bool = False):
    """--datasets: comma-separated dataset names (default: all 17). One Modal
    container per dataset, spawned concurrently.
    --modal-gpu: override GPU type (default: A10G). Pass "none" for CPU-only.
    --modal-cpu: override CPU core count (default: 4).
    --force: recompute and overwrite even if a cached .joblib already exists."""
    selected = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    fn = embed_atom_jepa
    options = {}
    if modal_gpu:
        options["gpu"] = None if modal_gpu.lower() == "none" else modal_gpu
    if modal_cpu:
        options["cpu"] = modal_cpu
    if options:
        fn = fn.with_options(**options)

    calls = {d: fn.spawn(d, force=force) for d in selected}
    for d, call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {d}: {e}")
