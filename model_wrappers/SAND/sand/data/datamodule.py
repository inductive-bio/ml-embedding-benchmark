"""Lightning datamodule and samplers for the SAND graph-pair training pipeline."""

from typing import Optional, cast, Dict, Union

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Sampler

from sand.mol_utils.graph import MolecularGraph, MolecularGraphCollection
import zarr


def _load_array(group: zarr.Group, path: str) -> np.ndarray:
    """Read a named sub-array of a zarr Group fully into memory."""
    return np.asarray(cast(zarr.Array, group[path])[:])


class BatchIndexSampler(Sampler):
    """Batch sampler that emits integer-index batches with optional DDP sharding.

    Supports chunk-based shuffling for very large datasets to avoid allocating
    a full index array up-front.
    """

    def __init__(
        self,
        length: int,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        shuffle_chunk_size: int = 100_000,
    ) -> None:
        """Initialise the sampler.

        Args:
            length (int): Total number of samples in the dataset.
            batch_size (int): Number of samples per batch.
            shuffle (bool): If ``True``, use chunk-based shuffling.
            drop_last (bool): Drop the last incomplete batch.
            shuffle_chunk_size (int): Chunk size used for the two-level
                shuffle.
        """
        self.length = length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.shuffle_chunk_size = shuffle_chunk_size

        # DDP-aware: shard indices across ranks if distributed
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def _get_indices(self) -> np.ndarray:
        """Return the subset of indices for this rank."""
        return np.arange(self.rank, self.length, self.world_size)

    def __iter__(self):
        """Yield batches of indices, sharded across DDP ranks."""
        indices = self._get_indices()
        if self.shuffle:
            # Chunk-based shuffling: shuffle chunk order, then shuffle within each chunk.
            # Avoids allocating a full index array for very large datasets.
            num_chunks = (len(indices) + self.shuffle_chunk_size - 1) // self.shuffle_chunk_size
            chunk_order = np.random.permutation(num_chunks)
            for chunk_idx in chunk_order:
                start = chunk_idx * self.shuffle_chunk_size
                end = min(start + self.shuffle_chunk_size, len(indices))
                chunk_indices = indices[start:end]
                chunk_indices = np.random.permutation(chunk_indices)
                for i in range(0, len(chunk_indices), self.batch_size):
                    batch = chunk_indices[i : i + self.batch_size]
                    if self.drop_last and len(batch) < self.batch_size:
                        continue
                    yield batch
        else:
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self) -> int:
        """Return the number of batches for this rank."""
        n = (self.length + self.world_size - 1) // self.world_size
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class SANDMolecularGraphDataset(Dataset):
    """Dataset of (query, match, score) graph pairs backed by a :class:`MolecularGraphCollection`."""

    def __init__(
        self,
        molecular_graph_collection: MolecularGraphCollection,
        query_idxs: np.ndarray,
        match_idxs: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        """Initialise the dataset.

        Args:
            molecular_graph_collection (MolecularGraphCollection): Flat store of
                pre-featurised molecular graphs.
            query_idxs (np.ndarray): 1-D integer array of query molecule
                indices.
            match_idxs (np.ndarray): 1-D integer array of match molecule
                indices.
            scores (np.ndarray): 1-D float array of similarity scores.
        """
        self.molecular_graph_collection = molecular_graph_collection
        self.query_idxs = query_idxs
        self.match_idxs = match_idxs
        self.scores = scores

    def __len__(self) -> int:
        return len(self.query_idxs)

    def __getitem__(self, idx: int) -> Dict[str, Union[MolecularGraph, torch.Tensor]]:
        """Return a single (query_graph, match_graph, score) sample.

        Args:
            idx (int): Sample index.

        Returns:
            Dict with keys ``"query_graph"``, ``"match_graph"``, and
            ``"score"`` (float tensor).
        """
        query_idx = self.query_idxs[idx]
        match_idx = self.match_idxs[idx]
        score = torch.tensor(self.scores[idx]).float()
        query_graph = self.molecular_graph_collection[query_idx]
        match_graph = self.molecular_graph_collection[match_idx]
        out = {
            "query_graph": query_graph,
            "match_graph": match_graph,
            "score": score,
        }
        return out

    def collate_fn(self, data_list) -> MolecularGraph:
        """Collate a list of samples into a single batched dict.

        Interleaves query and match graphs so the resulting batch tensor
        alternates query/match pairs.  Adds ``"num_graphs"``, ``"score"``,
        and ``"batch_size"`` entries to the returned dict.

        Args:
            data_list: List of dicts as returned by :meth:`__getitem__`.

        Returns:
            Batched dict with keys ``"batch"``, ``"score"``,
            ``"num_graphs"``, and ``"batch_size"``.
        """
        graphs = []
        for data in data_list:
            graphs.append(data["query_graph"])
            graphs.append(data["match_graph"])
        batch = MolecularGraph.from_graph_list(graphs)
        batch["num_graphs"] = batch["batch_size"]
        batch["score"] = torch.stack([data["score"] for data in data_list], dim=0)
        batch["batch_size"] = len(data_list)
        return batch


class SandDataModule(LightningDataModule):
    """Graph-pair datamodule backed by zarr stores.

    A single zarr graph store holds the molecular graphs (``x``, ``edge_index``,
    ``edge_attr``, ``node_offsets``, ``edge_offsets``). A second zarr score store
    holds the ``query_idx`` / ``match_idx`` pair indices and the target score for
    the ``train`` and ``val`` splits.

    The target score array is selected via ``score_key`` so the same datamodule
    can serve either ROCS shape scores (the default ``"score"``) or fingerprint
    similarity scores (e.g. ``"fp_score"``) without code changes. The default
    preserves the original behaviour so existing checkpoints/hparams load
    unchanged.
    """

    def __init__(
        self,
        graph_storage_path: str,
        score_storage_path: str,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        score_key: str = "score",
    ) -> None:
        """Initialise the datamodule.

        Args:
            graph_storage_path (str): Path to the zarr group containing
                molecular graph arrays (``x``, ``edge_index``, ``edge_attr``,
                ``node_offsets``, ``edge_offsets``).
            score_storage_path (str): Path to the zarr group containing score
                pair arrays (``train/`` and ``val/`` sub-groups).
            batch_size (int): Number of graph pairs per batch.
            num_workers (int): DataLoader worker processes.
            pin_memory (bool): Pin DataLoader output tensors to page-locked
                memory.
            persistent_workers (bool): Keep DataLoader workers alive between
                epochs.
            prefetch_factor (Optional[int]): Batches prefetched per worker.
            score_key (str): Sub-array name for the target score within each
                split group (e.g. ``"score"`` or ``"fp_score"``).
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.score_key = score_key
        # zarr.open() is typed to return Array | Group; both stores are always
        # opened as Groups containing named sub-arrays ("x", "train/query_idx", ...).
        self.graph_storage = cast(zarr.Group, zarr.open(graph_storage_path, mode="r"))
        self.score_storage = cast(zarr.Group, zarr.open(score_storage_path, mode="r"))

    def _mol_collection(self) -> MolecularGraphCollection:
        """Load all molecular graph arrays from zarr and return a collection."""
        return MolecularGraphCollection(
            x=_load_array(self.graph_storage, "x"),
            edge_index=_load_array(self.graph_storage, "edge_index"),
            edge_attr=_load_array(self.graph_storage, "edge_attr"),
            node_offsets=_load_array(self.graph_storage, "node_offsets"),
            edge_offsets=_load_array(self.graph_storage, "edge_offsets"),
        )

    def _get_dataloader(self, mode: str) -> DataLoader:
        """Build a DataLoader for the given split.

        Args:
            mode (str): Split name — ``"train"`` or ``"val"``.

        Returns:
            DataLoader yielding collated graph-pair batches.
        """
        ds = SANDMolecularGraphDataset(
            molecular_graph_collection=self._mol_collection(),
            query_idxs=_load_array(self.score_storage, f"{mode}/query_idx"),
            match_idxs=_load_array(self.score_storage, f"{mode}/match_idx"),
            scores=_load_array(self.score_storage, f"{mode}/{self.score_key}"),
        )
        return DataLoader(
            ds,
            batch_sampler=BatchIndexSampler(
                length=len(ds),
                batch_size=self.batch_size,
                shuffle=(mode == "train"),
            ),
            collate_fn=ds.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("val")
