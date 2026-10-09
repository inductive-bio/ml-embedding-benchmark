"""Zarr-store writers for SAND training data.

Two zarr stores are consumed by :class:`sand.data.datamodule.SandDataModule`:

**graphs.zarr** — flat CSR-style store of pre-featurised molecular graphs::

    x            shape (total_atoms, 16)   dtype uint8
    edge_index   shape (2, total_edges)    dtype int32
    edge_attr    shape (total_edges, 5)    dtype uint8
    node_offsets shape (n_mols + 1,)       dtype int64
    edge_offsets shape (n_mols + 1,)       dtype int64

**scores.zarr** — (query_idx, match_idx, score) pairs, split into
``train/`` and ``val/`` groups::

    train/query_idx  shape (n_train_pairs,)  dtype int32
    train/match_idx  shape (n_train_pairs,)  dtype int32
    train/score      shape (n_train_pairs,)  dtype float32
    val/…            same layout

``graphs.zarr`` is built here from a list of SMILES via
:func:`build_graph_store`. ``scores.zarr`` is built from a precomputed
``(query_smiles, match_smiles, score)`` dataset via :func:`append_score_pairs`.

Usage::

    from sand.data.writer import build_graph_store

    build_graph_store(smiles_list, "graphs.zarr")
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import zarr
from tqdm import tqdm

from sand.mol_utils.rdkit.molecule import RDMolecule


def _featurize_one(sml: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Featurise a single SMILES. Returns None on failure (caller warns)."""
    try:
        mol = RDMolecule.from_smiles(sml)
        g = mol.to_graph()
    except Exception:
        return None

    x = np.asarray(g.x, dtype=np.uint8)
    # edge_index produced by the featurizer has shape (2, E) when
    # permute_edge_index=True (the default).
    ei = np.asarray(g.edge_index, dtype=np.int32)
    ea = np.asarray(g.edge_attr, dtype=np.uint8)
    return x, ei, ea


def build_graph_store(
    smiles: List[str],
    out_zarr_path: Union[str, Path],
    chunk_size: int = 10_000,
    n_workers: Optional[int] = None,
) -> None:
    """Featurise a list of SMILES strings and write a flat CSR graph store.

    Invalid SMILES are skipped with a warning; the returned store preserves
    the original row order of valid SMILES only.

    Parameters
    ----------
    smiles:
        List of SMILES strings (one per molecule).
    out_zarr_path:
        Destination path for the zarr group (created fresh; overwrites any
        existing store at that path).
    chunk_size:
        Number of molecules processed between intermediate flushes.  Larger
        values are faster but use more peak memory.
    n_workers:
        Number of worker processes to featurise molecules in parallel.
        ``None`` or ``1`` runs single-process (original behaviour); results
        are always emitted in input order regardless of worker count, so
        the output row order/alignment with ``smiles`` is unaffected.
    """
    out_zarr_path = str(out_zarr_path)

    all_x: List[np.ndarray] = []
    all_ei: List[np.ndarray] = []
    all_ea: List[np.ndarray] = []
    node_offsets: List[int] = [0]
    edge_offsets: List[int] = [0]
    skipped = 0

    if n_workers is None or n_workers <= 1:
        for sml in tqdm(smiles, desc="Featurising molecules"):
            try:
                mol = RDMolecule.from_smiles(sml)
                g = mol.to_graph()
            except Exception as exc:
                warnings.warn(f"Skipping SMILES '{sml}': {exc}")
                skipped += 1
                continue

            x = np.asarray(g.x, dtype=np.uint8)
            ei = np.asarray(g.edge_index, dtype=np.int32)
            ea = np.asarray(g.edge_attr, dtype=np.uint8)

            all_x.append(x)
            all_ei.append(ei)
            all_ea.append(ea)
            node_offsets.append(node_offsets[-1] + x.shape[0])
            edge_offsets.append(edge_offsets[-1] + ea.shape[0])
    else:
        # executor.map returns results in input order (not imap_unordered),
        # so row order/alignment with `smiles` is preserved.
        chunksize = max(1, len(smiles) // (n_workers * 4))
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = executor.map(_featurize_one, smiles, chunksize=chunksize)
            for sml, result in tqdm(
                zip(smiles, results), total=len(smiles), desc="Featurising molecules"
            ):
                if result is None:
                    warnings.warn(f"Skipping SMILES '{sml}': featurisation failed")
                    skipped += 1
                    continue

                x, ei, ea = result
                all_x.append(x)
                all_ei.append(ei)
                all_ea.append(ea)
                node_offsets.append(node_offsets[-1] + x.shape[0])
                edge_offsets.append(edge_offsets[-1] + ea.shape[0])

    if skipped:
        warnings.warn(f"build_graph_store: skipped {skipped} invalid SMILES.")

    n_mols = len(node_offsets) - 1
    if n_mols == 0:
        raise ValueError("No valid molecules to write.")

    x_cat = np.concatenate(all_x, axis=0)                 # (total_atoms, 16)
    ei_cat = np.concatenate(all_ei, axis=1)                # (2, total_edges)
    ea_cat = np.concatenate(all_ea, axis=0)                # (total_edges, 5)
    no_arr = np.array(node_offsets, dtype=np.int64)
    eo_arr = np.array(edge_offsets, dtype=np.int64)

    root = zarr.open_group(out_zarr_path, mode="w")
    root.create_array(
        "x", data=x_cat,
        chunks=(min(100_000, len(x_cat)), x_cat.shape[1]),
    )
    root.create_array(
        "edge_index", data=ei_cat,
        chunks=(2, min(200_000, ei_cat.shape[1])),
    )
    root.create_array(
        "edge_attr", data=ea_cat,
        chunks=(min(200_000, len(ea_cat)), ea_cat.shape[1]),
    )
    root.create_array("node_offsets", data=no_arr)
    root.create_array("edge_offsets", data=eo_arr)

    print(
        f"build_graph_store: wrote {n_mols:,} graphs → {out_zarr_path}\n"
        f"  atoms: {len(x_cat):,}  edges: {ei_cat.shape[1]:,}"
    )


def append_score_pairs(
    scores_zarr_path: Union[str, Path],
    split: str,
    query_idx: np.ndarray,
    match_idx: np.ndarray,
    scores: np.ndarray,
) -> None:
    """Append a batch of (query_idx, match_idx, score) triples to a score store.

    The zarr group is created if it does not yet exist; arrays are grown
    via :meth:`zarr.Array.append`.

    Parameters
    ----------
    scores_zarr_path:
        Path to the score zarr group.
    split:
        Sub-group name: ``"train"`` or ``"val"``.
    query_idx, match_idx:
        1-D integer arrays of equal length.
    scores:
        1-D float array of the same length.
    """
    root = zarr.open_group(str(scores_zarr_path), mode="a")
    grp = root.require_group(split)

    q = np.asarray(query_idx, dtype=np.int32).ravel()
    m = np.asarray(match_idx, dtype=np.int32).ravel()
    s = np.asarray(scores, dtype=np.float32).ravel()

    chunk = (min(100_000, len(q)),)
    for name, data in [("query_idx", q), ("match_idx", m), ("score", s)]:
        if name in grp:
            grp[name].append(data)
        else:
            grp.create_array(name, data=data, chunks=chunk)
