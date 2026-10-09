"""Compute the model_wrappers/descriptors/ embeddings (rdkit2d, mordred2d, and
the 3 conformer-dependent families x 2 levels of theory) on Modal, fanned out
far below dataset granularity so hundreds of containers run at once instead of
one process working through a whole 45k-row corpus locally.

Grain is (dataset, model_name, batch of BATCH_SIZE molecules) -- conformer
generation (ETKDG, optionally MMFF94s) and Mordred's full 2D+3D descriptor set
are the slow steps here (measured locally: ~0.45s/molecule for
mordred3d_etkdg on real OpenADMET compounds), so a whole-dataset unit of work
would let the biggest datasets (ExpansionRx, up to ~7300 rows) dominate
wall-clock even with every other combo done. Small bounded batches also make
any single failure/preemption cheap to redo, same rationale as
scripts/modal_score_errors.py's per-(embedding,method,endpoint) grain.

Each batch call returns its embedding matrix directly (no Volume round trip
per batch); the local entrypoint concatenates batches in order and makes one
final call per (dataset, model_name) to assemble the EmbeddedDataset (needs
the prepared Dataset's splits/labels) and write it to the Volume -- matching
what embed.py does locally, just split across many containers.

Before running for the first time, data/prepared needs to be on the Volume
(already true as of this writing -- confirmed via `modal volume ls
ml-embedding-benchmark /prepared`). If it's ever missing:

    modal volume put ml-embedding-benchmark data/prepared /prepared

Usage:
    modal run scripts/modal_embed_descriptors.py
    modal run scripts/modal_embed_descriptors.py --datasets reg_openadmet_pxr_pEC50
    modal run scripts/modal_embed_descriptors.py --models rdkit2d,mordred2d
    modal run scripts/modal_embed_descriptors.py --batch-size 100

Run detached for a full run -- a non-detached `modal run` tears down the whole
app (killing every still-running container) if the local client disconnects,
same gotcha as every other Modal script in this repo:

    modal run --detach scripts/modal_embed_descriptors.py

After it completes, pull results back:

    modal volume get ml-embedding-benchmark /embedded data/embedded
"""

import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VOLUME_NAME = "ml-embedding-benchmark"
MOUNT = pathlib.Path("/data")
DEFAULT_BATCH_SIZE = 200

MODEL_NAMES = [
    "rdkit2d",
    "mordred2d",
    "rdkit3d_etkdg",
    "rdkit3d_mmff94s",
    "mordred3d_etkdg",
    "mordred3d_mmff94s",
    "mordred2d_rdkit3d_etkdg",
    "mordred2d_rdkit3d_mmff94s",
]

app = modal.App("ml-embedding-benchmark-descriptors")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",  # src/common/types.py imports it unconditionally (NumpyEncoder's torch.Tensor handling)
        "joblib", "pandas", "numpy", "rdkit", "mordredcommunity",
    )
    .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
    .add_local_file(PROJECT_ROOT / "config" / "illegal_smiles.txt", remote_path="/project/config/illegal_smiles.txt")
    .add_local_dir(
        PROJECT_ROOT / "model_wrappers" / "descriptors",
        remote_path="/project/model_wrappers/descriptors",
        ignore=["venv", "venv/**"],
    )
)


def _sys_path_setup():
    import sys
    for p in ("/project", "/project/model_wrappers/descriptors"):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_filtered_dataset(dataset_name: str):
    """Load data/prepared/<dataset_name>.joblib and apply the same
    filter_out_problematic_molecules step embed.py applies before embedding,
    so row count/order here matches every other embedder's output exactly --
    compute_prediction_errors.py aligns X to data/prepared by row position,
    not by molecule_id.
    """
    import joblib
    volume.reload()
    dataset = joblib.load(str(MOUNT / "prepared" / f"{dataset_name}.joblib"))
    with open("/project/config/illegal_smiles.txt") as f:
        illegal_smiles = [line.strip() for line in f.readlines()]
    dataset.filter_out_problematic_molecules(illegal_smiles)
    return dataset


def _dataset_num_rows(dataset_name: str) -> int:
    _sys_path_setup()
    return len(_load_filtered_dataset(dataset_name).data)


def _list_existing() -> list:
    """(dataset, model_name) stems already saved on the Volume, so a rerun
    only fills in what's missing (same idempotency contract as
    is_already_embedded() in src/embedding/embedding.py)."""
    volume.reload()
    existing = []
    embedded_dir = MOUNT / "embedded"
    if not embedded_dir.is_dir():
        return existing
    for dataset_dir in embedded_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        for f in dataset_dir.glob("*.joblib"):
            existing.append((dataset_dir.name, f.stem))
    return existing


def _embed_batch(dataset_name: str, model_name: str, batch_start: int, batch_size: int):
    _sys_path_setup()
    from wrapper import get_embedder

    dataset = _load_filtered_dataset(dataset_name)
    batch_df = dataset.data.iloc[batch_start:batch_start + batch_size]
    model = get_embedder(model_name)
    return model.embed(batch_df)  # np.ndarray (len(batch_df), dim), NaN-filled per-row on failure


def _merge_and_save(dataset_name: str, model_name: str, X_full):
    _sys_path_setup()
    from src.common.types import EmbeddedDataset
    import joblib
    import os

    dataset = _load_filtered_dataset(dataset_name)
    embedded = EmbeddedDataset(
        name=dataset.name, task=dataset.task, embedder=model_name,
        splits=dataset.splits, X=X_full, y=dataset.labels,
    )
    invalid = embedded.remove_failed_embeddings()

    out_dir = MOUNT / "embedded" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.joblib"
    joblib.dump(embedded, str(out_path))
    volume.commit()
    return f"ok  {dataset_name}/{model_name}  ({invalid} invalid dropped, final shape {embedded.X.shape})"


CPU = 2
# A batch's dominant cost is conformer generation (ETKDG, optionally MMFF94s)
# + Mordred's full descriptor DAG -- measured locally at up to ~0.45s/molecule
# for mordred3d_etkdg. 30 min covers a BATCH_SIZE=200 batch at ~9s/molecule,
# 20x the slowest rate seen so far.
BATCH_TIMEOUT = 30 * 60

row_count_fn = app.function(image=IMAGE, volumes={str(MOUNT): volume}, timeout=120)(_dataset_num_rows)
list_existing_fn = app.function(image=IMAGE, volumes={str(MOUNT): volume}, timeout=120)(_list_existing)
embed_batch_fn = app.function(image=IMAGE, volumes={str(MOUNT): volume}, timeout=BATCH_TIMEOUT, cpu=CPU)(_embed_batch)
merge_and_save_fn = app.function(image=IMAGE, volumes={str(MOUNT): volume}, timeout=600)(_merge_and_save)


@app.local_entrypoint()
def main(
    datasets: str = "reg_openadmet_*",
    models: str = ",".join(MODEL_NAMES),
    batch_size: int = DEFAULT_BATCH_SIZE,
    override: bool = False,
):
    """--datasets: glob (without .yaml) over config/dataset/ to select endpoints.
    --models: comma-separated model_name(s) from model_wrappers/descriptors/wrapper.py.
    --batch-size: molecules per container call.
    --override: recompute even if data/embedded/<dataset>/<model_name>.joblib
    already exists on the Volume."""
    import glob
    import os

    import numpy as np

    dataset_names = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(str(PROJECT_ROOT / "config" / "dataset" / f"{datasets}.yaml"))
    )
    model_names = [m.strip() for m in models.split(",") if m.strip()]
    unknown = set(model_names) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown model_name(s): {sorted(unknown)}. Known: {MODEL_NAMES}")

    print(f"{len(dataset_names)} dataset(s) x {len(model_names)} model(s), batch_size={batch_size}")

    existing = set() if override else set(list_existing_fn.remote())
    combos = [
        (d, m) for d in dataset_names for m in model_names
        if (d, m) not in existing
    ]
    skipped = len(dataset_names) * len(model_names) - len(combos)
    if skipped:
        print(f"Skipping {skipped} combo(s) already on the Volume (pass --override to recompute)")

    row_count_calls = {d: row_count_fn.spawn(d) for d in {d for d, _ in combos}}
    row_counts = {d: call.get() for d, call in row_count_calls.items()}

    # Spawn every batch for every combo up front -- Modal's own scheduler
    # handles concurrency/back-pressure across however many containers the
    # account allows, rather than us artificially serializing per combo.
    batch_calls = {}  # (dataset, model) -> [(batch_start, AsyncFunctionCall)]
    for dataset_name, model_name in combos:
        n = row_counts[dataset_name]
        starts = list(range(0, n, batch_size))
        batch_calls[(dataset_name, model_name)] = [
            (start, embed_batch_fn.spawn(dataset_name, model_name, start, batch_size))
            for start in starts
        ]

    total_batches = sum(len(v) for v in batch_calls.values())
    print(f"Fanning out {total_batches} batch container(s) across {len(combos)} (dataset, model) combo(s)")

    succeeded = 0
    failed = []
    for (dataset_name, model_name), calls in batch_calls.items():
        try:
            batch_rows = []
            for start, call in sorted(calls, key=lambda sc: sc[0]):
                batch_rows.append(call.get())
            X_full = np.concatenate(batch_rows, axis=0)
            print(merge_and_save_fn.remote(dataset_name, model_name, X_full))
            succeeded += 1
        except Exception as e:
            failed.append((f"{dataset_name}/{model_name}", str(e)))

    print(f"Done: {succeeded} succeeded, {len(failed)} failed")
    for key, err in failed:
        print(f"FAILED  {key}: {err}")
