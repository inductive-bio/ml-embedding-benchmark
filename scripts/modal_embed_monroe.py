"""Run embed.py on Modal for the `monroe` wrapper, one container per dataset.

    modal run scripts/modal_embed_monroe.py
    modal run scripts/modal_embed_monroe.py --datasets reg_openadmet_pxr_pEC50
    modal run scripts/modal_embed_monroe.py --modal-gpu A10G

Monroe (blazejba/Monroe) is a GRIT graph transformer pretrained on PM6 +
PCBA, genuinely 3D at inference (RDKit ETKDG+MMFF94s conformer positions feed
the encoder, same category as unimol -- see docs/embedding_expansion_plan.md's
"true-3D GNN candidates" section). Unlike minimol/KERMT, its encoder
(monroe/model/grit.py) only calls torch_geometric.utils.scatter -- no
torch-scatter/torch-sparse/torch-cluster compiled extensions, so there's no
exact (torch, cuda) wheel combo to pin against and the image is a plain
pip_install.

The released checkpoint ships in the repo via Git LFS (checkpoint/weights.pt,
~300MB) and is baked into the image at build time via add_local_dir below
(the real weights must already be pulled locally -- `git lfs pull` inside
model_wrappers/Monroe -- not left as an LFS pointer, or the build will bake in
a ~130-byte pointer file and every embed call will fail to torch.load it).

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

app = modal.App("ml-embedding-benchmark-monroe")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/Monroe"
    return (
        modal.Image.debian_slim(python_version="3.11")  # matches Monroe's own .python-version
        .apt_install("git")
        # datamol (imported directly by monroe/preprocessing/build_eval_graphs.py
        # and monroe/run_inference.py) pulls in rdkit.Chem.PandasTools ->
        # rdkit.Chem.Draw at import time, which needs X11 rendering libs
        # debian_slim doesn't ship by default -- same fix modal_embed_minimol.py
        # and modal_embed_unimol.py use for the same underlying import chain.
        .apt_install("libxrender1", "libxext6", "libsm6")
        .pip_install("torch==2.5.1", "torch_geometric==2.6.1")
        .pip_install(
            "rdkit", "datamol", "numpy", "pandas", "scipy", "scikit-learn",
            "tqdm", "pyarrow", "wandb", "hydra-core", "hydra-joblib-launcher", "joblib",
        )
        .add_local_dir(
            PROJECT_ROOT / "model_wrappers" / "Monroe",
            remote_path=wrapper_dir,
            # .git excluded: the nested repo's LFS pack duplicates the
            # ~300MB checkpoint blob, which would double the image's upload
            # size for no reason -- checkpoint/ (the real, already-pulled
            # weights) is included via the directory copy itself.
            ignore=["venv", "venv/**", ".git", ".git/**"],
            copy=True,
        )
        .run_commands(f"cd {wrapper_dir} && pip install -e .")
        # repo mount, after run_commands -- Modal forbids run_commands after a
        # non-copy add_local_dir.
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config", remote_path="/project/config")
        .add_local_file(PROJECT_ROOT / "embed.py", remote_path="/project/embed.py")
    )


def _run_embed_monroe(dataset_name: str, force: bool = False) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/Monroe"

    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "embedded").mkdir(parents=True, exist_ok=True)
    # Modal warm-starts/reuses containers across multiple .spawn() calls -- a
    # reused container already has these symlinks from its prior call, so
    # guard with lexists (not exists, which follows and can misbehave on a
    # dangling symlink) like every other recent modal_embed_*.py script.
    if not os.path.lexists(f"{project_root}/data/prepared"):
        os.symlink(str(MOUNT / "prepared"), f"{project_root}/data/prepared", target_is_directory=True)
    if not os.path.lexists(f"{project_root}/data/embedded"):
        os.symlink(str(MOUNT / "embedded"), f"{project_root}/data/embedded", target_is_directory=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{project_root}:{wrapper_dir}"

    cmd = [
        "python", "-u", "embed.py",
        "+experiment=monroe",
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

embed_monroe = app.function(
    name="embed_monroe",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    gpu="T4",
    cpu=4.0,
    timeout=60 * 30,
)(_run_embed_monroe)


@app.local_entrypoint()
def main(
    datasets: str = ",".join(DATASETS),
    modal_gpu: str = "",
    modal_cpu: float = 0,
    force: bool = False,
):
    """--datasets: comma-separated dataset names (default: all 17). One Modal
    container per dataset, spawned concurrently.
    --modal-gpu: override GPU type (default: T4). Pass "none" for CPU-only.
    --modal-cpu: override CPU core count (default: 4).
    --force: recompute and overwrite even if a cached .joblib already exists."""
    selected = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    fn = embed_monroe
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
