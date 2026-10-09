"""MMFF94s-optimize (UFF fallback) the ETKDG conformer ensembles that
scripts/modal_generate_conformers.py already wrote to the Volume, and write
the optimized coords + per-conformer energies to /conformers/<challenge>_mmff94s.joblib.

Reuses the exact same per-batch shards generation left behind under
/conformers/_batches/<challenge>/batch_*.joblib as the unit of work here too
-- each optimize container loads only its own small shard (~1MB) instead of
the full per-challenge merged file (up to 231MB for expansionrx), and the
batch boundaries (molecule groupings) are already fixed by that earlier run.

Molecules are reconstructed as bonded Chem.Mol objects from SMILES (not the
stored atomic_nums/coords alone, which have no bond information) with the
stored ETKDG coords set as conformers -- same AddHs()-on-a-freshly-parsed-mol
sequence create_etkdg_ensemble used originally, so atom ordering matches
exactly (verified live against a real sample before this was built at scale).

Usage:
    modal run --detach scripts/modal_optimize_conformers.py
    modal run --detach scripts/modal_optimize_conformers.py --challenges pxr

After it completes, pull results back:
    modal volume get ml-embedding-benchmark /conformers data/conformers
"""

import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VOLUME_NAME = "ml-embedding-benchmark"
MOUNT = pathlib.Path("/data")

CHALLENGES = ["expansionrx", "polaris_antiviral", "pxr"]
DEFAULT_PER_MOLECULE_TIMEOUT = 60

app = modal.App("ml-embedding-benchmark-conformers-opt")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("joblib", "numpy", "rdkit")
    .env({"PYTHONPATH": "/project"})
    .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
)


class _PerMoleculeTimeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _PerMoleculeTimeout("per-molecule optimization timeout")


def _list_batch_indices(challenge: str) -> list:
    volume.reload()
    batch_dir = MOUNT / "conformers" / "_batches" / challenge
    return sorted(int(p.stem.split("_")[1]) for p in batch_dir.glob("batch_*.joblib"))


def _optimize_batch(challenge: str, batch_idx: int, per_molecule_timeout: int) -> str:
    import signal

    import joblib
    import numpy as np
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    in_path = MOUNT / "conformers" / "_batches" / challenge / f"batch_{batch_idx:05d}.joblib"
    out_path = MOUNT / "conformers" / "_batches_mmff94s" / challenge / f"batch_{batch_idx:05d}.joblib"
    volume.reload()
    if out_path.exists():
        return f"skip (exists) {challenge}/batch_{batch_idx:05d}"

    from src.common.conformers import mmff_optimize

    raw = joblib.load(str(in_path))
    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    results = {}
    for smiles, entry in raw.items():
        if "error" in entry:
            results[smiles] = {"error": f"raw generation failed: {entry['error']}"}
            continue
        try:
            signal.alarm(per_molecule_timeout)
            try:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                if mol.GetNumAtoms() != entry["atomic_nums"].shape[0]:
                    raise ValueError("atom count mismatch reconstructing mol from SMILES")
                ensemble = []
                for coords in entry["coords"]:
                    m2 = Chem.Mol(mol)
                    conf = Chem.Conformer(m2.GetNumAtoms())
                    for a, (x, y, z) in enumerate(coords):
                        conf.SetAtomPosition(a, Point3D(float(x), float(y), float(z)))
                    m2.AddConformer(conf, assignId=True)
                    ensemble.append(m2)
                converged, energies = mmff_optimize(ensemble)
            finally:
                signal.alarm(0)
            if not converged:
                raise ValueError("no conformer converged during MMFF/UFF optimization")
            coords_stack = np.stack(
                [m.GetConformer().GetPositions().astype(np.float32) for m in converged], axis=0
            )
            results[smiles] = {
                "atomic_nums": entry["atomic_nums"],
                "coords": coords_stack,
                "energies": np.array(energies, dtype=np.float32),
                "formal_charge": entry["formal_charge"],
            }
        except Exception as e:
            results[smiles] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, str(out_path))
    volume.commit()
    n_ok = sum(1 for v in results.values() if "error" not in v)
    return f"ok {challenge}/batch_{batch_idx:05d} ({n_ok}/{len(raw)} succeeded)"


def _merge_challenge(challenge: str, batch_indices: list) -> str:
    import joblib

    volume.reload()
    batch_dir = MOUNT / "conformers" / "_batches_mmff94s" / challenge
    merged = {}
    missing = []
    for idx in batch_indices:
        shard_path = batch_dir / f"batch_{idx:05d}.joblib"
        if not shard_path.exists():
            missing.append(idx)
            continue
        merged.update(joblib.load(str(shard_path)))

    out_path = MOUNT / "conformers" / f"{challenge}_mmff94s.joblib"
    joblib.dump(merged, str(out_path))
    volume.commit()

    n_ok = sum(1 for v in merged.values() if "error" not in v)
    msg = (
        f"ok {challenge}: merged {len(merged)} molecules ({n_ok} succeeded, "
        f"{len(merged) - n_ok} failed) from {len(batch_indices) - len(missing)}/{len(batch_indices)} batches"
    )
    if missing:
        msg += f" -- MISSING batches: {missing}"
    return msg


LIST_CPU = 1
OPT_CPU = 1
MERGE_CPU = 2
OPT_TIMEOUT = 20 * 60

list_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=120, cpu=LIST_CPU)(_list_batch_indices)
optimize_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=OPT_TIMEOUT, cpu=OPT_CPU)(_optimize_batch)
merge_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=600, cpu=MERGE_CPU)(_merge_challenge)


@app.local_entrypoint()
def main(
    challenges: str = ",".join(CHALLENGES),
    per_molecule_timeout: int = DEFAULT_PER_MOLECULE_TIMEOUT,
):
    """--challenges: comma-separated subset of expansionrx,polaris_antiviral,pxr (default: all 3).
    --per-molecule-timeout: seconds before one molecule's optimization is abandoned
    and recorded as a timeout error, so it can't sink the rest of its batch."""
    selected = [c.strip() for c in challenges.split(",") if c.strip()]
    unknown = set(selected) - set(CHALLENGES)
    if unknown:
        raise ValueError(f"Unknown challenge(s): {sorted(unknown)}. Known: {CHALLENGES}")

    challenge_batch_indices = {}
    for challenge in selected:
        indices = list_fn.remote(challenge)
        challenge_batch_indices[challenge] = indices
        print(f"{challenge}: {len(indices)} batch(es) to optimize")

    total_batches = sum(len(v) for v in challenge_batch_indices.values())
    print(f"Fanning out {total_batches} batch container(s) across {len(selected)} challenge(s)...")

    calls = {
        (challenge, idx): optimize_fn.spawn(challenge, idx, per_molecule_timeout)
        for challenge, indices in challenge_batch_indices.items()
        for idx in indices
    }

    failed = []
    for key, call in calls.items():
        try:
            call.get()
        except Exception as e:
            failed.append((key, str(e)))
    print(f"Batch optimization done: {len(calls) - len(failed)} succeeded, {len(failed)} failed")
    for key, err in failed:
        print(f"FAILED  {key}: {err}")

    for challenge, indices in challenge_batch_indices.items():
        print(merge_fn.remote(challenge, indices))
