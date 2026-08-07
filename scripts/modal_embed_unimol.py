"""Run embed.py on Modal for just the `unimol` wrapper (both unimolv1/v2
variants -- config/experiment/unimol.yaml's own sweep already covers both, no
per-model fan-out needed), one container per dataset for parallelism.

    modal run scripts/modal_embed_unimol.py
    modal run scripts/modal_embed_unimol.py --datasets reg_openadmet_pxr_pEC50
    modal run scripts/modal_embed_unimol.py --modal-gpu A10G
    modal run scripts/modal_embed_unimol.py --modal-gpu none --modal-cpu 8

Split out from modal_embed.py: that script's `_build_embed_functions()`
decorates a Modal function per wrapper at import time, so even a
single-wrapper request there pays for building/verifying every non-excluded
wrapper's image (~8, 15-20+ min serially) before the requested wrapper's
function starts -- confirmed live in the 2026-08-06 session. This script only
ever builds one image.

Also fixes `unimol/wrapper.py`'s GPU handling -- it was previously hardcoded
to `use_cuda=False`, seemingly because it was written/tested on a machine
without a working CUDA driver. `unimol_tools`' own `use_cuda` handling already
falls back to CPU safely when CUDA isn't available (`torch.cuda.is_available()
and use_cuda`, confirmed by reading its installed source), so there was no
real reason to hardcode it off, and no reason not to request a GPU here.

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
    "reg_openadmet_expansionrx_LogS",
    "reg_openadmet_polaris_admet_HLM",
    "reg_openadmet_polaris_admet_KSOL",
    "reg_openadmet_polaris_admet_LogD",
    "reg_openadmet_polaris_admet_MDR1_MDCKII",
    "reg_openadmet_polaris_admet_MLM",
    "reg_openadmet_polaris_potency_pIC50_MERS_CoV_Mpro",
    "reg_openadmet_polaris_potency_pIC50_SARS_CoV_2_Mpro",
    "reg_openadmet_pxr_pEC50",
]

app = modal.App("ml-embedding-benchmark-unimol")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/unimol"
    return (
        modal.Image.debian_slim(python_version="3.10")  # matches unimol's own .python-version, no bump needed
        .apt_install("git")
        # unimol_tools imports rdkit.Chem.PandasTools -> ... -> rdkit.Chem.Draw
        # at import time (just for the side effect, no actual rendering happens)
        # which needs X11 rendering libs debian_slim doesn't ship by default --
        # confirmed live: `ImportError: libXrender.so.1: cannot open shared
        # object file`.
        .apt_install("libxrender1", "libxext6", "libsm6")
        .pip_install(
            # mirrors modal_embed.py's _install_base_requirements -- see that
            # function's docstring for why each of these is needed beyond
            # what unimol's own init.sh installs.
            "hydra-core", "hydra-joblib-launcher", "torch", "joblib", "tqdm",
            "pandas", "numpy<2.0.0,>=1.26.4", "rdkit", "setuptools<81",
        )
        .add_local_dir(PROJECT_ROOT / "model_wrappers" / "unimol", remote_path=wrapper_dir, copy=True)
        .run_commands(f"cd {wrapper_dir} && INSTALL_DEP=true bash -e init.sh")
        # init.sh's own `pip install unimol_tools --upgrade` pulls numpy back
        # past 2.0 regardless of the pin above -- re-assert it, same as the
        # local run recipe in docs/openadmet_migration_status.md does. Without
        # this, pandas (built against numpy 1.x) throws "numpy.dtype size
        # changed, may indicate binary incompatibility" at import time.
        .pip_install("numpy<2.0.0")
        # repo mount, after run_commands -- Modal forbids run_commands after a
        # non-copy add_local_dir.
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config", remote_path="/project/config")
        .add_local_file(PROJECT_ROOT / "embed.py", remote_path="/project/embed.py")
    )


def _run_embed_unimol(dataset_name: str) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/unimol"

    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "embedded").mkdir(parents=True, exist_ok=True)
    os.symlink(str(MOUNT / "prepared"), f"{project_root}/data/prepared", target_is_directory=True)
    os.symlink(str(MOUNT / "embedded"), f"{project_root}/data/embedded", target_is_directory=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{project_root}:{wrapper_dir}"

    cmd = [
        "python", "-u", "embed.py",
        "--multirun",
        "+experiment=unimol",
        f"+dataset={dataset_name}",
        "++cache=true",
    ]

    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"embed.py failed for dataset={dataset_name} (exit {result.returncode})")
    return f"ok  {dataset_name}"


IMAGE = build_image()

embed_unimol = app.function(
    name="embed_unimol",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    gpu="T4",
    cpu=4.0,  # RDKit conformer generation is CPU-bound and doesn't touch the GPU
    timeout=60 * 60 * 3,
)(_run_embed_unimol)


@app.local_entrypoint()
def main(datasets: str = ",".join(DATASETS), modal_gpu: str = "", modal_cpu: float = 0):
    """--datasets: comma-separated dataset names (default: all 17). One Modal
    container per dataset, spawned concurrently -- both unimolv1/v2 run inside
    each container.
    --modal-gpu: override GPU type (default: T4). Pass "none" for CPU-only.
    --modal-cpu: override CPU core count (default: 4)."""
    selected = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    fn = embed_unimol
    options = {}
    if modal_gpu:
        options["gpu"] = None if modal_gpu.lower() == "none" else modal_gpu
    if modal_cpu:
        options["cpu"] = modal_cpu
    if options:
        fn = fn.with_options(**options)

    calls = {d: fn.spawn(d) for d in selected}
    for d, call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {d}: {e}")
