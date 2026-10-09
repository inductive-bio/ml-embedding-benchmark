"""Boltzmann- (and, for rdkit3d only, mean-) aggregated 3D descriptor
embeddings, computed from the full MMFF94s-optimized conformer ensembles
(scripts/modal_generate_conformers.py + modal_optimize_conformers.py's
/conformers/_batches_mmff94s/<challenge>/batch_*.joblib), instead of a single
lowest-energy conformer like the existing rdkit3d_mmff94s/mordred3d_mmff94s
embedders use.

New model names (added alongside, not replacing, the existing 6 single-conformer
variants -- rdkit3d_etkdg, rdkit3d_mmff94s, mordred3d_etkdg, mordred3d_mmff94s,
mordred2d_rdkit3d_etkdg, mordred2d_rdkit3d_mmff94s):
  - rdkit3d_boltzmann            912-dim, energy-weighted average across the ensemble
  - rdkit3d_mean                 912-dim, plain average across the SAME ensemble
                                  (isolates aggregation method from Boltzmann,
                                  holding the conformer set fixed)
  - mordred3d_boltzmann          ~1826-dim, energy-weighted average
  - mordred2d_rdkit3d_boltzmann  mordred2d (fresh, no conformer needed) concatenated
                                  with rdkit3d_boltzmann

Boltzmann weights use kT at 298.15 K (~0.5925 kcal/mol, matching the kcal/mol
units RDKit's MMFF94s/UFF energies are already in), computed from the
molecule's own minimum energy for numerical stability (doesn't change the
normalized weights).

Two phases:
  1. Per-challenge, per-batch fan-out over the *same* batch shards the MMFF94s
     optimization step already wrote -- for every successfully-optimized
     molecule, compute all 4 target vectors from its conformer ensemble +
     energies. Merged into /conformers/<challenge>_aggregated.joblib (dict
     keyed by canonical SMILES).
  2. Per-dataset materialization: for each of the 17 OpenADMET datasets, look
     up its own SMILES (through the same filter_out_problematic_molecules
     preprocessing embed.py applies) against its challenge's aggregated dict,
     build each new model's EmbeddedDataset exactly like every other embedding
     in this repo (NaN-fill + remove_failed_embeddings()), and save to
     data/embedded/<dataset>/<model_name>.joblib.

Usage:
    modal run --detach scripts/modal_embed_aggregated_conformers.py
    modal run --detach scripts/modal_embed_aggregated_conformers.py --phase aggregate
    modal run --detach scripts/modal_embed_aggregated_conformers.py --phase materialize
"""

import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VOLUME_NAME = "ml-embedding-benchmark"
MOUNT = pathlib.Path("/data")

CHALLENGE_DATASET_GLOBS = {
    "expansionrx": ["reg_openadmet_expansionrx_*"],
    "polaris_antiviral": ["reg_openadmet_polaris_admet_*", "reg_openadmet_polaris_potency_*"],
    "pxr": ["reg_openadmet_pxr_*"],
}
NEW_MODEL_NAMES = ["rdkit3d_boltzmann", "rdkit3d_mean", "mordred3d_boltzmann", "mordred2d_rdkit3d_boltzmann"]
KT_298K = 1.987204e-3 * 298.15  # kcal/mol, matches RDKit MMFF94s/UFF energy units
# Same fix as model_wrappers/descriptors/wrapper.py's UNBOUNDED_MORDRED_DESCRIPTORS
# (duplicated rather than imported -- this image doesn't ship wrapper.py):
# these graph-theoretic descriptors are effectively unbounded and blew up
# Ridge's predictions by orders of magnitude on structurally unusual test
# compounds. Applied per-conformer, before Boltzmann/mean aggregation, so
# these variants stay consistent with mordred2d/mordred3d_etkdg/mmff94s.
UNBOUNDED_MORDRED_DESCRIPTORS = {"VR1_A", "VR2_A", "DetourIndex"}

app = modal.App("ml-embedding-benchmark-embed-aggregated")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",  # src/common/types.py imports it unconditionally
        "joblib", "pandas", "numpy", "rdkit", "mordredcommunity",
    )
    .env({"PYTHONPATH": "/project"})
    .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
    .add_local_file(PROJECT_ROOT / "config" / "illegal_smiles.txt", remote_path="/project/config/illegal_smiles.txt")
    .add_local_file(
        PROJECT_ROOT / "model_wrappers" / "descriptors" / "rdkit_3d.py",
        remote_path="/project/rdkit_3d.py",
    )
)


def _boltzmann_weights(energies):
    import numpy as np
    energies = np.asarray(energies, dtype=np.float64)
    rel = energies - energies.min()
    w = np.exp(-rel / KT_298K)
    return w / w.sum()


def _mol_with_coords(smiles, atomic_nums, coords):
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() != len(atomic_nums):
        raise ValueError("atom count mismatch reconstructing mol from SMILES")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    return mol


# ---------------------------------------------------------------------------
# Phase 1: aggregate over the MMFF94s-optimized ensemble, per unique molecule
# ---------------------------------------------------------------------------


def _list_optimized_batches(challenge: str) -> list:
    volume.reload()
    batch_dir = MOUNT / "conformers" / "_batches_mmff94s" / challenge
    return sorted(int(p.stem.split("_")[1]) for p in batch_dir.glob("batch_*.joblib"))


class _PerMoleculeTimeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _PerMoleculeTimeout("per-molecule aggregation timeout")


def _aggregate_batch(challenge: str, batch_idx: int, per_molecule_timeout: int = 180) -> str:
    import signal
    import sys
    sys.path.insert(0, "/project")  # rdkit_3d.py lives at /project/rdkit_3d.py

    import joblib
    import numpy as np
    from mordred import Calculator, descriptors as mordred_descriptors
    from rdkit import Chem
    from rdkit_3d import compute_rdkit_3d

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)

    in_path = MOUNT / "conformers" / "_batches_mmff94s" / challenge / f"batch_{batch_idx:05d}.joblib"
    out_path = MOUNT / "conformers" / "_batches_aggregated" / challenge / f"batch_{batch_idx:05d}.joblib"
    volume.reload()
    if out_path.exists():
        return f"skip (exists) {challenge}/batch_{batch_idx:05d}"

    calc2d = Calculator(mordred_descriptors, ignore_3D=True)
    calc3d = Calculator(mordred_descriptors, ignore_3D=False)
    unbounded_idx_2d = [i for i, d in enumerate(calc2d.descriptors) if str(d) in UNBOUNDED_MORDRED_DESCRIPTORS]
    unbounded_idx_3d = [i for i, d in enumerate(calc3d.descriptors) if str(d) in UNBOUNDED_MORDRED_DESCRIPTORS]

    optimized = joblib.load(str(in_path))
    results = {}
    for smiles, entry in optimized.items():
        row = {}
        # mordred2d needs no conformer -- compute it regardless of whether
        # this molecule's optimization succeeded, so mordred2d_rdkit3d_boltzmann
        # only drops a row when the 3D half actually failed.
        try:
            mol2d = Chem.MolFromSmiles(smiles)
            vec2d = np.nan_to_num(np.asarray(list(calc2d(mol2d).fill_missing()), dtype=np.float64), nan=0.0)
            vec2d[unbounded_idx_2d] = np.sign(vec2d[unbounded_idx_2d]) * np.log1p(np.abs(vec2d[unbounded_idx_2d]))
            row["mordred2d"] = vec2d
        except Exception as e:
            row["mordred2d"] = {"error": f"{type(e).__name__}: {e}"}

        if "error" in entry:
            row["rdkit3d_boltzmann"] = {"error": f"raw optimization failed: {entry['error']}"}
            row["rdkit3d_mean"] = row["rdkit3d_boltzmann"]
            row["mordred3d_boltzmann"] = row["rdkit3d_boltzmann"]
            results[smiles] = row
            continue

        try:
            signal.alarm(per_molecule_timeout)
            try:
                weights = _boltzmann_weights(entry["energies"])
                rdkit_vectors = []
                mordred_vectors = []
                for coords in entry["coords"]:
                    mol = _mol_with_coords(smiles, entry["atomic_nums"], coords)
                    rdkit_vectors.append(compute_rdkit_3d(mol))
                    mordred_vec = np.nan_to_num(np.asarray(list(calc3d(mol).fill_missing()), dtype=np.float64), nan=0.0)
                    mordred_vec[unbounded_idx_3d] = np.sign(mordred_vec[unbounded_idx_3d]) * np.log1p(np.abs(mordred_vec[unbounded_idx_3d]))
                    mordred_vectors.append(mordred_vec)
                rdkit_vectors = np.stack(rdkit_vectors, axis=0)
                mordred_vectors = np.stack(mordred_vectors, axis=0)
            finally:
                signal.alarm(0)

            row["rdkit3d_boltzmann"] = (weights[:, None] * rdkit_vectors).sum(axis=0)
            row["rdkit3d_mean"] = rdkit_vectors.mean(axis=0)
            row["mordred3d_boltzmann"] = (weights[:, None] * mordred_vectors).sum(axis=0)
        except Exception as e:
            err = {"error": f"{type(e).__name__}: {e}"}
            row["rdkit3d_boltzmann"] = err
            row["rdkit3d_mean"] = err
            row["mordred3d_boltzmann"] = err

        results[smiles] = row
    signal.signal(signal.SIGALRM, old_handler)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, str(out_path))
    volume.commit()
    n_ok = sum(1 for v in results.values() if not isinstance(v["rdkit3d_boltzmann"], dict))
    return f"ok {challenge}/batch_{batch_idx:05d} ({n_ok}/{len(optimized)} succeeded)"


def _merge_aggregated(challenge: str, batch_indices: list) -> str:
    import joblib

    volume.reload()
    batch_dir = MOUNT / "conformers" / "_batches_aggregated" / challenge
    merged = {}
    missing = []
    for idx in batch_indices:
        shard_path = batch_dir / f"batch_{idx:05d}.joblib"
        if not shard_path.exists():
            missing.append(idx)
            continue
        merged.update(joblib.load(str(shard_path)))

    out_path = MOUNT / "conformers" / f"{challenge}_aggregated.joblib"
    joblib.dump(merged, str(out_path))
    volume.commit()

    n_ok = sum(1 for v in merged.values() if not isinstance(v["rdkit3d_boltzmann"], dict))
    msg = (
        f"ok {challenge}: merged {len(merged)} molecules ({n_ok} with usable 3D, "
        f"{len(merged) - n_ok} failed) from {len(batch_indices) - len(missing)}/{len(batch_indices)} batches"
    )
    if missing:
        msg += f" -- MISSING batches: {missing}"
    return msg


# ---------------------------------------------------------------------------
# Phase 2: materialize per-dataset EmbeddedDataset files for each new model
# ---------------------------------------------------------------------------


def _materialize_dataset(dataset_name: str, challenge: str) -> str:
    import numpy as np
    import joblib
    from src.common.types import EmbeddedDataset

    volume.reload()
    dataset = joblib.load(str(MOUNT / "prepared" / f"{dataset_name}.joblib"))
    with open("/project/config/illegal_smiles.txt") as f:
        illegal_smiles = [line.strip() for line in f.readlines()]
    dataset.filter_out_problematic_molecules(illegal_smiles)

    aggregated = joblib.load(str(MOUNT / "conformers" / f"{challenge}_aggregated.joblib"))
    smiles_list = dataset.data["smiles"].tolist()

    def row_or_nan(key, dim_hint):
        rows = []
        dim = dim_hint
        for smi in smiles_list:
            entry = aggregated.get(smi)
            if entry is None or isinstance(entry[key], dict):
                rows.append(None)
            else:
                v = entry[key]
                dim = len(v)
                rows.append(v)
        return np.array([r if r is not None else np.full(dim, np.nan) for r in rows], dtype=np.float32)

    rdkit3d_boltzmann = row_or_nan("rdkit3d_boltzmann", 912)
    rdkit3d_mean = row_or_nan("rdkit3d_mean", 912)
    mordred3d_boltzmann = row_or_nan("mordred3d_boltzmann", 1826)
    mordred2d = row_or_nan("mordred2d", 1613)
    mordred2d_rdkit3d_boltzmann = np.concatenate([mordred2d, rdkit3d_boltzmann], axis=1)

    outputs = {
        "rdkit3d_boltzmann": rdkit3d_boltzmann,
        "rdkit3d_mean": rdkit3d_mean,
        "mordred3d_boltzmann": mordred3d_boltzmann,
        "mordred2d_rdkit3d_boltzmann": mordred2d_rdkit3d_boltzmann,
    }

    out_dir = MOUNT / "embedded" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    msgs = []
    for model_name, X in outputs.items():
        embedded = EmbeddedDataset(
            name=dataset.name, task=dataset.task, embedder=model_name,
            splits=dataset.splits, X=X, y=dataset.labels,
        )
        invalid = embedded.remove_failed_embeddings()
        joblib.dump(embedded, str(out_dir / f"{model_name}.joblib"))
        msgs.append(f"{model_name} ({invalid} invalid dropped, shape {embedded.X.shape})")
    volume.commit()
    return f"ok {dataset_name}: " + "; ".join(msgs)


AGG_CPU = 1
AGG_TIMEOUT = 30 * 60
MATERIALIZE_CPU = 2

list_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=120, cpu=1)(_list_optimized_batches)
aggregate_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=AGG_TIMEOUT, cpu=AGG_CPU)(_aggregate_batch)
merge_agg_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=600, cpu=2)(_merge_aggregated)
materialize_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=600, cpu=MATERIALIZE_CPU)(_materialize_dataset)


@app.local_entrypoint()
def main(challenges: str = ",".join(CHALLENGE_DATASET_GLOBS), phase: str = "both"):
    """--challenges: comma-separated subset of expansionrx,polaris_antiviral,pxr.
    --phase: 'aggregate' (phase 1 only), 'materialize' (phase 2 only, needs
    phase 1 already done), or 'both' (default)."""
    import glob
    import os

    selected = [c.strip() for c in challenges.split(",") if c.strip()]
    unknown = set(selected) - set(CHALLENGE_DATASET_GLOBS)
    if unknown:
        raise ValueError(f"Unknown challenge(s): {sorted(unknown)}. Known: {list(CHALLENGE_DATASET_GLOBS)}")

    if phase in ("aggregate", "both"):
        challenge_batch_indices = {}
        for challenge in selected:
            indices = list_fn.remote(challenge)
            challenge_batch_indices[challenge] = indices
            print(f"{challenge}: {len(indices)} batch(es) to aggregate")

        calls = {
            (challenge, idx): aggregate_fn.spawn(challenge, idx)
            for challenge, indices in challenge_batch_indices.items()
            for idx in indices
        }
        print(f"Fanning out {len(calls)} aggregation batch container(s)...")
        failed = []
        for key, call in calls.items():
            try:
                call.get()
            except Exception as e:
                failed.append((key, str(e)))
        print(f"Aggregation done: {len(calls) - len(failed)} succeeded, {len(failed)} failed")
        for key, err in failed:
            print(f"FAILED  {key}: {err}")

        for challenge, indices in challenge_batch_indices.items():
            print(merge_agg_fn.remote(challenge, indices))

    if phase in ("materialize", "both"):
        dataset_to_challenge = {}
        for challenge, patterns in CHALLENGE_DATASET_GLOBS.items():
            if challenge not in selected:
                continue
            for pattern in patterns:
                for path in glob.glob(str(PROJECT_ROOT / "config" / "dataset" / f"{pattern}.yaml")):
                    dataset_to_challenge[os.path.splitext(os.path.basename(path))[0]] = challenge

        print(f"Materializing {len(dataset_to_challenge)} dataset(s)...")
        calls = {
            dataset_name: materialize_fn.spawn(dataset_name, challenge)
            for dataset_name, challenge in dataset_to_challenge.items()
        }
        for dataset_name, call in calls.items():
            try:
                print(call.get())
            except Exception as e:
                print(f"FAILED  {dataset_name}: {e}")
