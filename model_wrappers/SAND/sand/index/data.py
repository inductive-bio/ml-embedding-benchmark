"""HDF5 and Lance data stores and PyTorch Datasets for large-scale SMILES databases."""

import os
import shutil
from typing import Union, List, Optional, Dict, Literal, cast, TYPE_CHECKING
from contextlib import contextmanager
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from sand.mol_utils.graph import MolecularGraph

from torch.utils.data import Dataset, DataLoader, Sampler

if TYPE_CHECKING:
    import h5py


# Default columns to read from the Enamine REAL cxsmiles bz2 dumps.
ENAIME_ROWS = [
    "smiles",
    "id",
    "MW",
    "HAC",
    "sLogP",
    "HBA",
    "HBD",
    "RotBonds",
    "FSP3",
    "TPSA",
]


class HDF5Store:
    def __init__(self, file_path: str, offset: int = 0) -> None:
        """Initialise the HDF5 store.

        Args:
            file_path: Path to the HDF5 file.
            offset (int): Row offset applied to every index operation.
        """
        self.file_path = str(file_path)
        self.key = "data/table"
        self.offset = offset

    @contextmanager
    def _open(self):
        """Context manager for HDF5 file access."""
        import hdf5plugin  # noqa: F401  (registers HDF5 compression filters)
        import h5py

        with h5py.File(self.file_path, "r") as hdf:
            yield hdf

    def __len__(self) -> int:
        """Return the number of rows in the store (minus the offset)."""
        with self._open() as hdf:
            dataset = cast("h5py.Dataset", hdf[self.key])
            return dataset.shape[0]

    def _to_dataframe(self, data: np.ndarray) -> pd.DataFrame:
        """Convert HDF5 data to pandas DataFrame."""
        if data.ndim == 0:
            data = np.array([data])
        df = pd.DataFrame(data)
        # Decode byte strings
        for col in df.select_dtypes(include=[object]).columns:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
        return df

    def __getitem__(self, idx) -> pd.DataFrame:
        """Return one or more rows as a DataFrame.

        Args:
            idx: Integer, slice, list of ints, or numpy array of ints.

        Returns:
            pandas.DataFrame with the selected rows.
        """
        with self._open() as hdf:
            dataset = cast("h5py.Dataset", hdf[self.key])

            if isinstance(idx, int):
                data = dataset[idx + self.offset]
            elif isinstance(idx, slice):
                start = (idx.start or 0) + self.offset
                stop = (idx.stop or len(self)) + self.offset
                data = dataset[start:stop]
            elif isinstance(idx, (list, np.ndarray)):
                # h5py fancy indexing requires strictly increasing (sorted,
                # unique) indices, so deduplicate first and reconstruct the
                # original order (with any duplicates) via the inverse map.
                idx_array = np.array(idx)
                unique_idx, inverse = np.unique(idx_array, return_inverse=True)

                adjusted = unique_idx + self.offset
                data = dataset[adjusted]

                # Restore original order (and duplicate entries)
                data = data[inverse]
            else:
                raise TypeError(f"Index type {type(idx)} not supported")

            return self._to_dataframe(data)

    @classmethod
    def from_bz2(
        cls,
        bz2_path: str,
        output_path: Optional[str] = None,
        cols: List[str] = ENAIME_ROWS,
        min_itemsize: Optional[Dict[str, int]] = None,
        chunk_size: int = 1_000_000,
        complib: Literal["zlib", "lzo", "bzip2", "blosc"] = "blosc",
        complevel: int = 5,
    ):
        """
        Convert a bz2 file to HDF5 format with chunked processing.

        Args:
            bz2_path: Path to the input bz2 file
            output_path: Path for output HDF5 file (default: replaces .bz2 with .h5)
            cols: List of columns to include (default: ENAIME_ROWS)
            min_itemsize: Dictionary specifying minimum string lengths for columns
            chunk_size: Number of rows to process at once
            complib: Compression library for HDF5 (default: 'blosc')
            complevel: Compression level (default: 5)

        Returns:
            HDF5Store instance for the created file
        """
        # Default min_itemsize if not specified
        if min_itemsize is None:
            min_itemsize = {
                "smiles": 150,  # SMILES can be quite long for large molecules
                "id": 50,  # Adjust based on your ID format
            }

        # Define output path if not specified
        if output_path is None:
            output_path = bz2_path.replace(".bz2", ".h5")

        # Remove existing HDF5 file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)

        # Process in chunks and write to HDF5
        with pd.HDFStore(
            output_path, mode="w", complib=complib, complevel=complevel
        ) as store:
            chunk_iter = pd.read_table(
                bz2_path,
                compression="bz2",
                chunksize=chunk_size,
                iterator=True,
                usecols=cols,
            )

            for i, chunk in tqdm(enumerate(chunk_iter), desc="Converting bz2 to HDF5"):
                # Append each chunk to the HDF5 file
                store.append(
                    "data",
                    chunk,
                    format="table",
                    data_columns=True,
                    min_itemsize=min_itemsize,
                )

        print(f"Conversion complete! HDF5 file saved to: {output_path}")

        # Return an instance of HDF5Store for the created file
        return cls(output_path)


class InMemoryHDF5Store:
    def __init__(self, hdf_store: HDF5Store, limit: int = -1) -> None:
        """Load the HDF5 store into memory.

        Args:
            hdf_store: Source :class:`HDF5Store` to read from.
            limit (int): Maximum number of rows to load. ``-1`` loads all.
        """
        df = hdf_store[:limit]  # offset is already handled by hdf_store
        self.data = df

    def __getitem__(self, idx: Union[int, slice, List[int], np.ndarray]) -> pd.DataFrame:
        """Return one or more rows from the in-memory DataFrame.

        Args:
            idx: Integer (returns a single-row DataFrame) or any
                pandas-compatible iloc indexer for multi-row selection.

        Returns:
            pandas.DataFrame with the selected rows.
        """
        if isinstance(idx, int):
            series = self.data.iloc[idx]
            # make a dataframe with a single row
            return pd.DataFrame([series])
        else:
            return self.data.iloc[idx]

    def __len__(self) -> int:
        return len(self.data)


class HDF5Dataset(Dataset):
    def __init__(
        self,
        hdf5_file_path: str,
        offset: int = 0,
        limit: int = -1,
        in_memory: bool = False,
    ) -> None:
        """Initialise the dataset.

        Args:
            hdf5_file_path (str): Path to the HDF5 file.
            offset (int): Row offset within the HDF5 store.
            limit (int): Maximum number of rows to use; ``-1`` uses all rows.
            in_memory (bool): If ``True``, load the entire dataset into RAM.
        """
        store = HDF5Store(hdf5_file_path, offset=offset)
        assert 0 <= offset < len(store), "Offset out of bounds"
        assert limit == -1 or limit > 0, "Limit must be -1 or positive"
        if in_memory:
            self.store = InMemoryHDF5Store(store, limit=limit)
        else:
            self.store = store
        self.offset = offset
        self.limit = limit

    def __len__(self) -> int:
        return len(self.store)

    def smiles_to_graph(self, smiles: str) -> MolecularGraph:
        """Convert a SMILES string to a :class:`MolecularGraph`.

        Args:
            smiles (str): SMILES string of the molecule.

        Returns:
            MolecularGraph with node/edge features and the original SMILES
            stored under ``graph["smiles"]``.
        """
        from sand.mol_utils.rdkit.molecule import RDMolecule

        mol = RDMolecule.from_smiles(smiles)
        graph = mol.to_graph(add_pos=False)
        graph["smiles"] = smiles
        return graph

    def __getitem__(
        self, idx: Union[int, slice, List[int], np.ndarray]
    ) -> Union[MolecularGraph, List[MolecularGraph]]:
        """Return a graph or list of graphs for the given index.

        Args:
            idx: Integer, slice, list, or numpy array of indices.

        Returns:
            A single :class:`MolecularGraph` for integer ``idx``, or a list of
            :class:`MolecularGraph` objects otherwise.
        """
        df = self.store[idx]
        if isinstance(idx, int):
            graph = self.smiles_to_graph(df.iloc[0].smiles)
            # have to  it with "" to avoid using pandas build in index col...
            graph["index"] = df["index"][0]
        else:
            graph_list = []
            for index, row in df.iterrows():
                graph = self.smiles_to_graph(row.smiles)
                graph["index"] = index
                graph_list.append(graph)
            graph = graph_list
        return graph

    def to_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        chunked: bool = True,
        **dl_kwargs,
    ) -> DataLoader:
        """Wrap this dataset in a :class:`~torch.utils.data.DataLoader`.

        Args:
            batch_size (int): Number of molecules per batch.
            shuffle (bool): Whether to shuffle the data. Must be ``False``
                when ``chunked=True``.
            num_workers (int): Number of DataLoader worker processes.
            chunked (bool): If ``True``, use :class:`ChunkedSampler` for
                sequential HDF5 access.
            **dl_kwargs: Additional keyword arguments forwarded to
                :class:`~torch.utils.data.DataLoader`.

        Returns:
            DataLoader yielding collated :class:`MolecularGraph` batches.
        """
        if chunked and shuffle:
            raise ValueError("If chunked is True, shuffle must be False")

        if chunked:
            batch_sampler = ChunkedSampler(self, batch_size)
            return DataLoader(
                self,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=MolecularGraph.from_graph_list,
                **dl_kwargs,
            )
        else:
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=False,
                collate_fn=MolecularGraph.from_graph_list,
                **dl_kwargs,
            )


class ChunkedSampler(Sampler):
    """Sampler that yields sequential chunks of indices for efficient HDF5 reading."""

    def __init__(self, dataset, chunk_size: int) -> None:
        """Initialise the sampler.

        Args:
            dataset: Dataset whose length determines the total number of items.
            chunk_size (int): Number of consecutive indices per batch.
        """
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.total_size = len(dataset)

    def __iter__(self):
        for start in range(0, self.total_size, self.chunk_size):
            end = min(start + self.chunk_size, self.total_size)
            yield list(range(start, end))
        # Ensure the last chunk is not empty
        # if start < self.total_size:
        #     yield list(range(start, self.total_size))

    def __len__(self) -> int:
        return (self.total_size + self.chunk_size - 1) // self.chunk_size


def no_opt(x):
    """Return the first element of ``x``."""
    return x[0]


class LanceStore:
    def __init__(self, file_path: str, offset: int = 0, limit: Optional[int] = None) -> None:
        """Initialise the Lance store.

        Args:
            file_path: Path to the Lance dataset directory.
            offset (int): Row offset applied to every index operation.
            limit (Optional[int]): Maximum number of rows to expose;
                ``None`` exposes all rows after the offset.
        """
        self.file_path = str(file_path)
        self.offset = offset
        self.limit = limit
        self._ds = None

    @property
    def ds(self):
        """Lazily opened Lance dataset handle, one per process."""
        # Lance is not fork-safe (its native dataset handle owns background
        # threads), so we never share one opened in a parent process across
        # a fork. Instead each process (main or DataLoader worker) lazily
        # opens its own handle the first time it's needed.
        if self._ds is None:
            import lance

            self._ds = lance.dataset(self.file_path)
        return self._ds

    def __len__(self) -> int:
        """Return the number of accessible rows (respecting offset and limit)."""
        length = len(self.ds) - self.offset
        if self.limit:
            length = min(self.limit, length)
        return length

    def __getitem__(self, idx) -> pd.DataFrame:
        """Return one or more rows as a DataFrame with a ``db_idx`` column.

        Args:
            idx: Integer, slice, numpy array, or list of ints.

        Returns:
            pandas.DataFrame with the requested rows and a ``db_idx`` column
            containing the absolute row indices in the Lance dataset.
        """
        if isinstance(idx, int):
            idx = [idx + self.offset]
        elif isinstance(idx, slice):
            start = (idx.start or 0) + self.offset
            stop = (idx.stop or len(self)) + self.offset
            idx = [i for i in range(start, stop)]
        elif isinstance(idx, np.ndarray):
            idx = idx + self.offset
        elif isinstance(idx, list):
            idx = np.array(idx) + self.offset
        else:
            raise TypeError(f"Index type {type(idx)} not supported")

        tbl = self.ds.take(idx)
        df = tbl.to_pandas()
        df["db_idx"] = idx
        return df

    @classmethod
    def from_bz2(
        cls,
        bz2_path: str,
        output_path: Optional[str] = None,
        cols: List[str] = ENAIME_ROWS,
        chunk_size: int = 1_000_000,
    ):
        """
        Convert a bz2 file to a Lance dataset with chunked processing.

        Args:
            bz2_path: Path to the input bz2 file
            output_path: Path for output Lance dataset (default: replaces .bz2 with .lance)
            cols: List of columns to include (default: ENAIME_ROWS)
            chunk_size: Number of rows to process at once

        Returns:
            LanceStore instance for the created dataset
        """
        import lance

        # Define output path if not specified
        if output_path is None:
            output_path = bz2_path.replace(".bz2", ".lance")

        # Remove existing Lance dataset if it exists (Lance datasets are directories)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        chunk_iter = pd.read_table(
            bz2_path,
            compression="bz2",
            chunksize=chunk_size,
            iterator=True,
            usecols=cols,
        )

        for i, chunk in tqdm(enumerate(chunk_iter), desc="Converting bz2 to Lance"):
            mode = "create" if i == 0 else "append"
            lance.write_dataset(chunk, output_path, mode=mode)

        print(f"Conversion complete! Lance dataset saved to: {output_path}")

        # Return an instance of LanceStore for the created dataset
        return cls(output_path)


class LanceDataset(Dataset):
    def __init__(self, file_path: str, offset: int = 0, limit: Optional[int] = None) -> None:
        """Initialise the Lance-backed dataset.

        Args:
            file_path: Path to the Lance dataset directory.
            offset (int): Row offset within the Lance store.
            limit (Optional[int]): Maximum number of rows; ``None`` uses all.
        """
        self.store = LanceStore(file_path, offset, limit)

    def __len__(self) -> int:
        return len(self.store)

    def sml_to_graph(self, sml: str) -> MolecularGraph:
        """Convert a SMILES string to a :class:`MolecularGraph`.

        Args:
            sml (str): SMILES string.

        Returns:
            MolecularGraph with node/edge features and the SMILES stored
            under ``graph["smiles"]``.
        """
        from sand.mol_utils.rdkit.molecule import RDMolecule

        mol = RDMolecule.from_smiles(sml)
        graph = mol.to_graph()
        graph["smiles"] = sml
        return graph

    def __getitem__(self, idx: Union[int, slice, np.ndarray, List[int]]) -> MolecularGraph:
        """Return a single graph or a collated batch of graphs.

        Args:
            idx: Index or slice passed to the underlying :class:`LanceStore`.

        Returns:
            A collated :class:`MolecularGraph` if multiple molecules are
            retrieved, or a single :class:`MolecularGraph` otherwise.
        """
        df = self.store[idx]
        smls = df.smiles.tolist()
        db_idx = df.db_idx.tolist()
        graph_list = []
        for db_idx_, sml in zip(db_idx, smls):
            graph = self.sml_to_graph(sml)
            graph["db_idx"] = torch.LongTensor([db_idx_])
            graph_list.append(graph)
        if len(graph_list) > 1:
            graph = MolecularGraph.from_graph_list(graph_list)
        else:
            graph = graph_list[0]
        return graph

    def to_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        chunked: bool = False,
        **dl_kwargs,
    ) -> DataLoader:
        if chunked and shuffle:
            raise ValueError("If chunked is True, shuffle must be False")

        if num_workers > 0:
            dl_kwargs.setdefault("multiprocessing_context", "spawn")

        if chunked:
            batch_sampler = ChunkedSampler(self, batch_size)
            return DataLoader(
                self,
                sampler=batch_sampler,
                num_workers=num_workers,
                # collate_fn=MolecularGraph.from_graph_list,
                collate_fn=no_opt,
                **dl_kwargs,
            )
        else:
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=False,
                collate_fn=MolecularGraph.from_graph_list,
                **dl_kwargs,
            )


def build_lance_store_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Convert an Enamine REAL cxsmiles .bz2 dump into a Lance dataset "
            "usable as a SAND search database."
        )
    )
    parser.add_argument(
        "bz2_path",
        help="Path to the input Enamine REAL .cxsmiles.bz2 file.",
    )
    parser.add_argument(
        "output_path",
        help="Path for the output Lance dataset (a directory ending in .lance).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows to process per chunk (default: 100000).",
    )
    args = parser.parse_args()

    LanceStore.from_bz2(
        bz2_path=args.bz2_path,
        output_path=args.output_path,
        chunk_size=args.chunk_size,
    )

