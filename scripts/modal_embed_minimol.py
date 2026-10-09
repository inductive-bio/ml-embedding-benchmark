"""Run embed.py on Modal for the `minimol` wrapper, one container per dataset.

    modal run scripts/modal_embed_minimol.py
    modal run scripts/modal_embed_minimol.py --datasets reg_openadmet_polaris_admet_HLM
    modal run scripts/modal_embed_minimol.py --modal-gpu A10G

minimol (graphcore-research/minimol) is a Graphium-based GNN featurizer,
cloned locally into model_wrappers/minimol/ with its real checkpoint restored
under minimol/ckpts/minimol_v1/ (the upstream repo ships it via git-lfs; the
checkout here has the real ~20MB state_dict.pth, not the LFS pointer). Its
torch-scatter/torch-sparse/torch-cluster deps only ship prebuilt wheels per
exact (torch, cuda) combo, so torch is pinned to the combo minimol's own
README validates against (2.3.0+cu121) rather than left to resolve freely.

Before running for the first time, upload data/prepared to the Volume (same
one modal_embed.py uses) if not already there:

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

app = modal.App("ml-embedding-benchmark-minimol")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/minimol"
    return (
        modal.Image.debian_slim(python_version="3.10")  # matches minimol's own .python-version
        .apt_install("git")
        # graphium's datamol dep imports rdkit.Chem.PandasTools -> ... ->
        # rdkit.Chem.Draw at import time (just for the side effect, no actual
        # rendering happens), which needs X11 rendering libs debian_slim
        # doesn't ship by default -- same fix modal_embed_unimol.py uses.
        .apt_install("libxrender1", "libxext6", "libsm6")
        .pip_install("torch==2.3.0", "torchvision==0.18.0", index_url="https://download.pytorch.org/whl/cu121")
        .pip_install(
            "torch-scatter", "torch-sparse", "torch-cluster",
            find_links="https://data.pyg.org/whl/torch-2.3.0+cu121.html",
        )
        .pip_install(
            # torchvision pinned explicitly: mup (a graphium dep) requires it
            # unpinned, which previously let pip upgrade past the pinned
            # torch==2.3.0 and broke the torch_scatter/torch_sparse .so files
            # built against 2.3.0+cu121 ("undefined symbol" at import time).
            # scipy pinned <1.13: graphium's featurizer builds its fake graph
            # (used to infer input dims) with an unrequested float16 dtype (an
            # upstream bug) -- scipy's coo_matrix silently accepted this
            # through 1.11.x but raises "scipy.sparse does not support dtype
            # float16" from ~1.13 on.
            "graphium==2.4.7", "torchvision==0.18.0", "hydra-core", "hydra-joblib-launcher",
            "joblib", "tqdm", "pandas", "numpy<2.0.0,>=1.26.4", "scipy==1.11.4", "rdkit", "setuptools<81",
        )
        .add_local_dir(
            PROJECT_ROOT / "model_wrappers" / "minimol",
            remote_path=wrapper_dir,
            ignore=["venv", "venv/**", ".git", ".git/**"],
            copy=True,
        )
        .run_commands(f"cd {wrapper_dir} && pip install -e .")
        # numpy<2.0.0 pin above can get pulled back past 2.0 by graphium's own
        # deps -- re-assert it after everything else installs, same pattern
        # modal_embed_unimol.py uses for the same reason (pandas built
        # against numpy 1.x throws "numpy.dtype size changed" at import time
        # against a numpy 2.x runtime).
        .pip_install("numpy<2.0.0")
        # repo mount, after run_commands -- Modal forbids run_commands after a
        # non-copy add_local_dir.
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config", remote_path="/project/config")
        .add_local_file(PROJECT_ROOT / "embed.py", remote_path="/project/embed.py")
    )


def _run_embed_minimol(dataset_name: str, force: bool = False) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/minimol"

    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "embedded").mkdir(parents=True, exist_ok=True)
    if not os.path.exists(f"{project_root}/data/prepared"):
        os.symlink(str(MOUNT / "prepared"), f"{project_root}/data/prepared", target_is_directory=True)
    if not os.path.exists(f"{project_root}/data/embedded"):
        os.symlink(str(MOUNT / "embedded"), f"{project_root}/data/embedded", target_is_directory=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{project_root}:{wrapper_dir}"

    cmd = [
        "python", "-u", "embed.py",
        "--multirun",
        "+experiment=minimol",
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

embed_minimol = app.function(
    name="embed_minimol",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    gpu="T4",
    cpu=4.0,
    timeout=60 * 60 * 3,
)(_run_embed_minimol)


@app.local_entrypoint()
def main(datasets: str = ",".join(DATASETS), modal_gpu: str = "", modal_cpu: float = 0, force: bool = False):
    """--datasets: comma-separated dataset names (default: all 17). One Modal
    container per dataset, spawned concurrently.
    --modal-gpu: override GPU type (default: T4). Pass "none" for CPU-only.
    --modal-cpu: override CPU core count (default: 4).
    --force: recompute and overwrite even if a cached .joblib already exists."""
    selected = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    fn = embed_minimol
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
