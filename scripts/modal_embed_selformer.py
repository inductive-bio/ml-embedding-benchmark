"""Run embed.py on Modal for just the `huggingface` wrapper's SELFormer/
SELFormer-Lite variants, one container per dataset for parallelism.

    modal run scripts/modal_embed_selformer.py
    modal run scripts/modal_embed_selformer.py --datasets reg_openadmet_pxr_pEC50
    modal run scripts/modal_embed_selformer.py --modal-gpu A10G

Split out from modal_embed.py for the same reason scripts/modal_embed_unimol.py
was: that script decorates a Modal function per wrapper at import time, so even
a single-wrapper request pays for building/verifying every non-excluded
wrapper's image first. This script only ever builds one image.

Also split out from the main `huggingface` sweep (rather than just adding
`selformer` back into its model glob) to avoid the unresolved GPU-memory issue
that broke that sweep's other variants on Modal T4 (see docs/
openadmet_migration_status.md) -- that sweep loads ~10 different models
sequentially in one long-running process without releasing GPU memory between
them. One container per dataset here means each container only ever loads
these 2 (small) models, once, then exits -- structurally avoids the same
failure mode regardless of whether it's fully understood.

SELFormer's weights (`SELFormer.zip`/`SELFormer-Lite.zip`) needed an
interactive Google Drive download -- already done this session (`gdown`,
same trick as `cddd`/`simson`) and extracted into
model_wrappers/huggingface/model_weights/{SELFormer,SELFormer-Lite}/, which
this script bakes into the image alongside the rest of the wrapper dir.

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
MODEL_GLOB = "glob(huggingface*selformer*)"

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

app = modal.App("ml-embedding-benchmark-selformer")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_image() -> modal.Image:
    wrapper_dir = "/project/model_wrappers/huggingface"
    return (
        modal.Image.debian_slim(python_version="3.10")  # matches huggingface's own .python-version
        .apt_install("git")
        .pip_install(
            # mirrors modal_embed.py's _install_base_requirements
            "hydra-core", "hydra-joblib-launcher", "torch", "joblib", "tqdm",
            "pandas", "numpy<2.0.0,>=1.26.4", "rdkit", "setuptools<81",
        )
        # transformers pinned to 4.46.3, same as the local huggingface venv --
        # SELFormer (unlike MolFormer) doesn't need transformers>=5.0, so no
        # reason to diverge from the version everything else there already uses.
        .pip_install(
            "transformers[torch]==4.46.3", "selfies==2.1.1", "sentencepiece",
            "protobuf", "sentence-transformers",
        )
        .add_local_dir(
            PROJECT_ROOT / "model_wrappers" / "huggingface",
            remote_path=wrapper_dir,
            copy=True,
            # venvs are local-only artifacts (multiple GB, irrelevant in the
            # container, which gets its own via pip_install above) -- exclude
            # them or the image build has to upload all of it for nothing.
            ignore=["venv", "venv/**", "venv_molformer", "venv_molformer/**"],
        )
        # repo mount, after the copy=True step above -- Modal forbids
        # run_commands after a non-copy add_local_dir (no run_commands needed
        # here since there's no init.sh step beyond what's already installed).
        .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
        .add_local_dir(PROJECT_ROOT / "config", remote_path="/project/config")
        .add_local_file(PROJECT_ROOT / "embed.py", remote_path="/project/embed.py")
    )


def _run_embed_selformer(dataset_name: str) -> str:
    import os
    import subprocess

    project_root = "/project"
    wrapper_dir = f"{project_root}/model_wrappers/huggingface"

    os.makedirs(f"{project_root}/data", exist_ok=True)
    (MOUNT / "embedded").mkdir(parents=True, exist_ok=True)
    os.symlink(str(MOUNT / "prepared"), f"{project_root}/data/prepared", target_is_directory=True)
    os.symlink(str(MOUNT / "embedded"), f"{project_root}/data/embedded", target_is_directory=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{project_root}:{wrapper_dir}"

    cmd = [
        "python", "-u", "embed.py",
        "--multirun",
        "+experiment=huggingface",
        f"+dataset={dataset_name}",
        f"+model={MODEL_GLOB}",
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

embed_selformer = app.function(
    name="embed_selformer",
    image=IMAGE,
    volumes={str(MOUNT): volume},
    gpu="T4",
    timeout=60 * 60 * 3,
)(_run_embed_selformer)


@app.local_entrypoint()
def main(datasets: str = ",".join(DATASETS), modal_gpu: str = ""):
    """--datasets: comma-separated dataset names (default: all 17). One Modal
    container per dataset, spawned concurrently -- both SELFormer/SELFormer-Lite
    run inside each container.
    --modal-gpu: override GPU type (default: T4). Pass "none" for CPU-only."""
    selected = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = set(selected) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {DATASETS}")

    fn = embed_selformer
    if modal_gpu:
        fn = fn.with_options(gpu=None if modal_gpu.lower() == "none" else modal_gpu)

    calls = {d: fn.spawn(d) for d in selected}
    for d, call in calls.items():
        try:
            print(call.get())
        except Exception as e:
            print(f"FAILED  {d}: {e}")
