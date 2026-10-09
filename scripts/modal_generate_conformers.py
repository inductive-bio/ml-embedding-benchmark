"""Generate ETKDG conformer ensembles (up to 50/molecule, RMS-pruned at 0.1 A,
no force-field optimization) for every unique molecule in the 3 OpenADMET
challenges, and write them to the Modal Volume under /conformers.

Deduplication is per-challenge, not global: the 17 endpoint datasets group
into exactly 3 challenges (docs/data_sources/{expansionrx,polaris_antiviral,
pxr}.md) -- ExpansionRx's 9 endpoints, Polaris antiviral's 7 (5 ADMET + 2
Potency), and PXR's 1. Within a challenge, endpoints overwhelmingly share the
same compound library (ExpansionRx: 36,003 rows -> 7,608 unique SMILES);
across challenges there's essentially no overlap (13,756 summed per-challenge
unique vs. 13,755 if deduped globally across all 17 datasets). Grouping by
challenge means an endpoint's conformers are always a lookup into its own
challenge's file, without needing to know about the other challenges.

Fan-out is at (challenge, batch-of-25-molecules) granularity -- per-molecule
cost is mostly ~1-1.5s (measured locally) but with rare outliers (the largest
of 13,755 SMILES took ~167s for 50 conformers), so small batches keep any one
container's worst case bounded rather than let an outlier stall a big batch.
Each batch worker writes its own shard directly to the Volume (not returned
over RPC) so a retry only ever recomputes that one batch; a final per-challenge
merge combines all shards into one /conformers/<challenge>.joblib.

Usage:
    modal run --detach scripts/modal_generate_conformers.py
    modal run --detach scripts/modal_generate_conformers.py --challenges pxr
    modal run --detach scripts/modal_generate_conformers.py --batch-size 10

After it completes, pull results back:
    modal volume get ml-embedding-benchmark /conformers data/conformers
"""

import pathlib

import modal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VOLUME_NAME = "ml-embedding-benchmark"
MOUNT = pathlib.Path("/data")

DEFAULT_NUM_CONFS = 50
DEFAULT_RMS_THRESHOLD = 0.1
DEFAULT_BATCH_SIZE = 25

# challenge -> glob pattern(s) over config/dataset/ / data/prepared/ names
CHALLENGES = {
    "expansionrx": ["reg_openadmet_expansionrx_*"],
    "polaris_antiviral": ["reg_openadmet_polaris_admet_*", "reg_openadmet_polaris_potency_*"],
    "pxr": ["reg_openadmet_pxr_*"],
}

app = modal.App("ml-embedding-benchmark-conformers")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",  # src/common/types.py imports it unconditionally (Dataset needs the class importable to unpickle)
        "joblib", "pandas", "numpy", "rdkit",
    )
    # Must precede add_local_* -- Modal requires build steps like .env() to
    # come before add_local_dir/add_local_file, or the build fails outright.
    .env({"PYTHONPATH": "/project"})
    .add_local_dir(PROJECT_ROOT / "src", remote_path="/project/src")
    .add_local_file(PROJECT_ROOT / "config" / "illegal_smiles.txt", remote_path="/project/config/illegal_smiles.txt")
)


def _extract_challenge_smiles(dataset_globs: list) -> list:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    import joblib

    volume.reload()
    with open("/project/config/illegal_smiles.txt") as f:
        illegal_smiles = {line.strip() for line in f.readlines()}

    unique = set()
    prepared_dir = MOUNT / "prepared"
    for pattern in dataset_globs:
        for path in sorted(prepared_dir.glob(f"{pattern}.joblib")):
            dataset = joblib.load(str(path))
            for smi in dataset.data["smiles"]:
                if smi in illegal_smiles:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                unique.add(Chem.MolToSmiles(mol))
    return sorted(unique)


class _PerMoleculeTimeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _PerMoleculeTimeout("per-molecule generation timeout")


def _generate_batch(
    challenge: str, batch_idx: int, smiles_batch: list, num_confs: int, rms_threshold: float,
    per_molecule_timeout: int = 180,
) -> str:
    import signal

    import numpy as np
    import joblib
    from rdkit import Chem

    shard_path = MOUNT / "conformers" / "_batches" / challenge / f"batch_{batch_idx:05d}.joblib"
    volume.reload()
    if shard_path.exists():
        return f"skip (exists) {challenge}/batch_{batch_idx:05d}"

    from src.common.conformers import create_etkdg_ensemble

    # A single pathologically large/flexible molecule can run 50-conformer
    # ETKDG far longer than the rest of the batch combined -- confirmed live:
    # 2 batches in a real run each hit the *container's* 1200s timeout on one
    # molecule, losing the other ~24 already-computed molecules in the same
    # batch too, since the shard is only written once at the end. Bounding
    # each molecule individually (SIGALRM -- confirmed RDKit's embedding loop
    # does check for pending signals mid-call, since Modal's own container
    # cancellation was seen interrupting it cleanly) keeps one bad molecule
    # from sinking its batch; it just gets recorded as a timeout error.
    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    results = {}
    for smiles in smiles_batch:
        try:
            signal.alarm(per_molecule_timeout)
            try:
                ensemble = create_etkdg_ensemble(smiles, num_confs=num_confs, rms_threshold=rms_threshold)
            finally:
                signal.alarm(0)
            if not ensemble:
                raise ValueError("ETKDG embedding produced no conformers")
            # Every conformer in the ensemble is a deep copy of the same
            # Hs-added mol (see create_etkdg_ensemble) -- atom composition and
            # formal charge are identical across all of them, only positions differ.
            atomic_nums = np.array([atom.GetAtomicNum() for atom in ensemble[0].GetAtoms()], dtype=np.int8)
            formal_charge = int(Chem.rdmolops.GetFormalCharge(ensemble[0]))
            coords_stack = np.stack(
                [m.GetConformer().GetPositions().astype(np.float32) for m in ensemble], axis=0
            )
            results[smiles] = {
                "atomic_nums": atomic_nums,
                "coords": coords_stack,
                "formal_charge": formal_charge,
            }
        except Exception as e:
            results[smiles] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, str(shard_path))
    volume.commit()
    n_ok = sum(1 for v in results.values() if "error" not in v)
    return f"ok {challenge}/batch_{batch_idx:05d} ({n_ok}/{len(smiles_batch)} succeeded)"


def _merge_challenge(challenge: str, num_batches: int) -> str:
    import joblib

    volume.reload()
    batch_dir = MOUNT / "conformers" / "_batches" / challenge
    merged = {}
    missing = []
    for idx in range(num_batches):
        shard_path = batch_dir / f"batch_{idx:05d}.joblib"
        if not shard_path.exists():
            missing.append(idx)
            continue
        merged.update(joblib.load(str(shard_path)))

    out_path = MOUNT / "conformers" / f"{challenge}.joblib"
    joblib.dump(merged, str(out_path))
    volume.commit()

    n_ok = sum(1 for v in merged.values() if "error" not in v)
    msg = f"ok {challenge}: merged {len(merged)} molecules ({n_ok} succeeded, {len(merged) - n_ok} failed) from {num_batches - len(missing)}/{num_batches} batches"
    if missing:
        msg += f" -- MISSING batches: {missing}"
    return msg


EXTRACT_CPU = 2
GEN_CPU = 1
MERGE_CPU = 2
DEFAULT_PER_MOLECULE_TIMEOUT = 180
# Headroom above batch_size * per_molecule_timeout's pathological worst case
# (every molecule in the batch timing out back to back) rather than the
# typical case, so the container timeout itself is never what kills a batch.
GEN_TIMEOUT = 60 * 60

extract_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=300, cpu=EXTRACT_CPU)(_extract_challenge_smiles)
generate_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=GEN_TIMEOUT, cpu=GEN_CPU)(_generate_batch)
merge_fn = app.function(image=image, volumes={str(MOUNT): volume}, timeout=600, cpu=MERGE_CPU)(_merge_challenge)


@app.local_entrypoint()
def main(
    challenges: str = ",".join(CHALLENGES),
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_confs: int = DEFAULT_NUM_CONFS,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
    per_molecule_timeout: int = DEFAULT_PER_MOLECULE_TIMEOUT,
):
    """--challenges: comma-separated subset of expansionrx,polaris_antiviral,pxr (default: all 3).
    --batch-size: molecules per container call.
    --num-confs / --rms-threshold: ETKDG ensemble generation parameters.
    --per-molecule-timeout: seconds before one molecule's generation is abandoned
    and recorded as a timeout error, so it can't sink the rest of its batch."""
    selected = [c.strip() for c in challenges.split(",") if c.strip()]
    unknown = set(selected) - set(CHALLENGES)
    if unknown:
        raise ValueError(f"Unknown challenge(s): {sorted(unknown)}. Known: {sorted(CHALLENGES)}")

    challenge_batches = {}
    for challenge in selected:
        smiles_list = extract_fn.remote(CHALLENGES[challenge])
        batches = [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]
        challenge_batches[challenge] = batches
        print(f"{challenge}: {len(smiles_list)} unique molecules, {len(batches)} batch(es)")

    total_batches = sum(len(b) for b in challenge_batches.values())
    print(f"Fanning out {total_batches} batch container(s) across {len(selected)} challenge(s)...")

    calls = {
        (challenge, idx): generate_fn.spawn(challenge, idx, batch, num_confs, rms_threshold, per_molecule_timeout)
        for challenge, batches in challenge_batches.items()
        for idx, batch in enumerate(batches)
    }

    failed = []
    for key, call in calls.items():
        try:
            call.get()
        except Exception as e:
            failed.append((key, str(e)))
    print(f"Batch generation done: {len(calls) - len(failed)} succeeded, {len(failed)} failed")
    for key, err in failed:
        print(f"FAILED  {key}: {err}")

    for challenge, batches in challenge_batches.items():
        print(merge_fn.remote(challenge, len(batches)))
