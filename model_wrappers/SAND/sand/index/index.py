"""FAISS-backed IVFPQ similarity index for SAND molecular embeddings."""

from typing import Union, List, Optional, Tuple
from contextlib import nullcontext
import torch
import faiss
from tqdm import tqdm
import numpy as np
from faiss.contrib import torch_utils
from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)

_AMP_DTYPES = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}


from sand.model import SANDModel

from sand.index.data import HDF5Store, LanceStore, LanceDataset


def path_to_str(path: Union[str, Path]) -> str:
    """Convert a :class:`~pathlib.Path` to a plain string; pass strings through unchanged.

    Args:
        path: A :class:`~pathlib.Path` or string.

    Returns:
        String representation of ``path``.
    """
    if isinstance(path, Path):
        return str(path)
    return path


class SANDIndex:
    """FAISS IVFPQ similarity index built on top of a :class:`SANDModel`.

    Manages the FAISS index lifecycle (build, add, search, save, load) and
    optionally backs searches against an HDF5 or Lance molecule database.
    """

    def __init__(
        self,
        model: SANDModel,
        n_probe: int = 512,
        index_on_gpu: bool = False,
        model_on_gpu: bool = False,
        db_file_path: Optional[Union[str, Path]] = None,
        index_file_path: Optional[Union[str, Path]] = None,
        mmapped: bool = False,
        f16_codes: bool = False,
        shard_index_multi_gpu: bool = False,
        compile_encoder: bool = False,
        amp_dtype: Optional[str] = None,
        matmul_precision: Optional[str] = None,
        overwrite_index: bool = False,
        num_workers: int = 0,
    ):
        """Initialise the index, loading or building the FAISS index as needed.

        If ``index_file_path`` already exists (and ``overwrite_index`` is
        ``False``) the index is loaded from disk; otherwise a new IVFPQ index
        is built from the model's trained quantizer and, if ``db_file_path``
        is provided, the entire database is indexed.

        Args:
            model (SANDModel): Trained SAND model.
            n_probe (int): Number of IVF cells to search at query time.
            index_on_gpu (bool): Move the FAISS index to GPU.
            model_on_gpu (bool): Move the encoder model to GPU.
            db_file_path: Path to the molecule database (``.h5`` or
                ``.lance``). ``None`` for an empty index.
            index_file_path: Path for saving/loading the FAISS index. ``None``
                keeps the index in memory only.
            mmapped (bool): Open the index file as a memory-mapped file.
            f16_codes (bool): Use float16 PQ codes on GPU for reduced memory.
            shard_index_multi_gpu (bool): Shard the index across all available
                GPUs.
            compile_encoder (bool): Compile the GNN encoder with
                ``torch.compile``.
            amp_dtype: Automatic mixed-precision dtype; one of ``"none"``,
                ``"bf16"``, ``"fp16"``, or ``None``.
            matmul_precision: Override
                ``torch.set_float32_matmul_precision``.
            overwrite_index (bool): If ``True``, rebuild the index even when an
                existing file is found at ``index_file_path``.
            num_workers (int): DataLoader workers for database indexing.
        """
        self.model_on_gpu = model_on_gpu
        self.index_on_gpu = index_on_gpu
        self.f16_codes = f16_codes
        self.model = model
        self.db_file_path = path_to_str(db_file_path) if db_file_path is not None else None
        if self.model.device.type == "cpu" and model_on_gpu:
            self.model = self.model.to("cuda")
        self.n_probe = n_probe
        self.shard_index_multi_gpu = shard_index_multi_gpu
        self.overwrite_index = overwrite_index
        self.num_workers = num_workers

        if matmul_precision is not None:
            logger.info(f"Setting torch matmul precision to {matmul_precision}.")
            torch.set_float32_matmul_precision(matmul_precision)

        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)
        self.compile_encoder = compile_encoder
        if self.compile_encoder:
            logger.info(f"Compiling encoder...")
            self._compile_encoder()

        self.init_store()

        if index_file_path is not None:
            if Path(index_file_path).exists() and not self.overwrite_index:
                logger.info(f"Loading index from {index_file_path} ...")
                self.load_from_disk(index_file_path, mmapped=mmapped)
            elif Path(index_file_path).exists() and self.overwrite_index:
                if db_file_path is None:
                    logger.info(f"Overriding index at {index_file_path} with empty FAISS index...")
                    self.build_faiss_index()
                    self.save_to_disk(index_file_path)
                elif (db_file_path is not None) and (not Path(db_file_path).exists()):
                    raise ValueError(
                        f"Database file {db_file_path} does not exist."
                    )
                else:
                    logger.info(f"Overriding index at {index_file_path}...")
                    self.build_faiss_index()
                    self._index_database(num_workers=self.num_workers)
                    self.save_to_disk(index_file_path)
            else:
                if (db_file_path is not None) and (not Path(db_file_path).exists()):
                    raise ValueError(
                        f"Database file {db_file_path} does not exist."
                    )

                elif (db_file_path is not None) and Path(db_file_path).exists():
                    logger.info(f"Index file {index_file_path} does not exist. Building new FAISS index for Database {db_file_path}...")
                    self.build_faiss_index()
                    self._index_database(num_workers=self.num_workers)
                    self.save_to_disk(index_file_path)
                elif (db_file_path is None):
                    logger.info(f"Index file {index_file_path} does not exist. No database provided. Building empty FAISS index and saving to disk...")
                    self.build_faiss_index()
                    self.save_to_disk(index_file_path)

        else:
            if (db_file_path is not None) and (not Path(db_file_path).exists()):
                raise ValueError(
                    f"Database file {db_file_path} does not exist"
                )
            elif (db_file_path is not None) and Path(db_file_path).exists():
                logger.info(f"Building FAISS index for Database {db_file_path}...")
                self.build_faiss_index()
                self._index_database(num_workers=self.num_workers)
            else:
                logger.info(f"Building empty FAISS index...")
                self.build_faiss_index()

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: Optional[str]) -> Optional[torch.dtype]:
        """Resolve an AMP dtype string to a :class:`torch.dtype`.

        Args:
            amp_dtype: One of ``"none"``, ``"bf16"``, ``"fp16"``, or ``None``.

        Returns:
            Corresponding :class:`torch.dtype`, or ``None`` for no AMP.
        """
        if amp_dtype is None:
            return None
        if amp_dtype not in _AMP_DTYPES:
            raise ValueError(
                f"amp_dtype must be one of {list(_AMP_DTYPES)}, got {amp_dtype!r}."
            )
        return _AMP_DTYPES[amp_dtype]

    def _compile_encoder(self) -> None:
        """Compile the backbone GNN encoder with ``torch.compile``."""
        self.model.backbone.encoder.model.gnn = torch.compile(
            self.model.backbone.encoder.model.gnn, dynamic=True
        )

    def autocast_context(self):
        """Return an autocast context manager for the configured AMP dtype.

        Returns a ``nullcontext`` when AMP is disabled.
        """
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast("cuda", dtype=self.amp_dtype)

    def init_store(self) -> None:
        """Initialise the molecule data store from ``db_file_path``.

        Sets ``self.store`` to an :class:`~sand.index.data.HDF5Store`,
        :class:`~sand.index.data.LanceStore`, or ``None`` depending on the
        file extension.
        """
        if self.db_file_path is None:
            self.store = None
        elif self.db_file_path.endswith(".h5"):
            self.store = HDF5Store(self.db_file_path)
        elif self.db_file_path.endswith(".lance"):
            self.store = LanceStore(self.db_file_path)
        else:
            raise ValueError(f"{self.db_file_path} not supported. Only .h5 and .lance files are supported.")

    def __len__(self) -> int:
        return self.index.ntotal

    def build_faiss_index(self) -> None:
        """Build the FAISS IVFPQ index from the model's trained quantizer."""
        trained_quantizer = self.model.backbone.ivfpq
        self._build_faiss_index_ivfpq(trained_quantizer)

    def _build_faiss_index_ivfpq(self, trained_quantizer) -> None:
        """Build a FAISS IVFPQ index from an IVFPQ quantizer."""
        n_list = (
            trained_quantizer.ivf_codebook_size**trained_quantizer.num_ivf_quantizer
        )
        print(f"Building FAISS index with {n_list} IVF lists.")
        m = trained_quantizer.num_pq_quantizer
        dim = trained_quantizer.input_dim
        ivf_code_book, pq_code_book = trained_quantizer.extract_code_books()
        self.ivf_code_book = ivf_code_book
        quantizer = faiss.IndexFlatIP(dim)
        quantizer.add(ivf_code_book)
        self.index = faiss.IndexIVFPQ(
            quantizer, dim, n_list, m, 8
        )  # 8 bits per subquantizer
        self.index.nprobe = self.n_probe
        faiss.copy_array_to_vector(pq_code_book.ravel(), self.index.pq.centroids)
        self.index.precompute_table()
        self.index.is_trained = True

        if self.index_on_gpu:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.f16_codes
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index, co)

    def add(
        self,
        emb: Union[np.ndarray, torch.Tensor],
        ids: Optional[Union[np.ndarray, torch.Tensor, List[int]]] = None,
    ):
        """Add pre-computed embeddings to the FAISS index.

        Args:
            emb: Embedding matrix of shape ``(N, D)``.
            ids: Optional 1-D array of integer IDs to assign to the vectors.
                If ``None``, FAISS assigns sequential IDs.
        """
        if isinstance(emb, torch.Tensor) and not self.index_on_gpu:
            emb = emb.cpu().numpy()
            if ids is not None and isinstance(ids, torch.Tensor):
                ids = ids.cpu().numpy()
        elif isinstance(emb, torch.Tensor) and (ids is not None):
            if isinstance(ids, list):
                ids = torch.LongTensor(ids)
            if isinstance(ids, np.ndarray):
                ids = torch.from_numpy(ids)
        if ids is not None:
            self.index.add_with_ids(emb, ids)
        else:
            self.index.add(emb)

    def encode_batch(self, batch, emb_key: str = None, batch_key: str = "batch") -> torch.Tensor:
        """Encode a batch dict to embeddings using the backbone model.

        Args:
            batch: Batch dictionary as produced by the dataloader collate
                function.
            emb_key (str): Key under which the model stores embeddings in the
                batch dict.
            batch_key (str): Key that holds the batch index tensor.

        Returns:
            Embedding tensor of shape ``(N, D)``.
        """
        with self.autocast_context():
            emb = self.model.encode_batch(batch, emb_key=emb_key, batch_key=batch_key)
        return emb

    def add_batch(
        self,
        batch,
        emb_key: str = "emb",
        batch_key: str = "batch",
        id_key: Optional[str] = None,
    ):
        """Encode a batch and add the resulting embeddings to the index.

        Args:
            batch: Batch dictionary.
            emb_key (str): Key for embeddings in the batch dict.
            batch_key (str): Key for the batch index tensor.
            id_key (Optional[str]): Key for vector IDs in the batch dict.
                ``None`` uses sequential FAISS IDs.

        Returns:
            Tuple of ``(embeddings, ids)`` where ``ids`` is ``None`` when no
            ``id_key`` is supplied.
        """
        emb = self.encode_batch(batch, emb_key=emb_key, batch_key=batch_key)
        if id_key is not None:
            ids = batch[id_key]
        else:
            ids = None
        self.add(emb, ids=ids)
        return emb, ids

    def add_from_dataloader(
        self,
        dataloader,
        emb_key: str = "emb",
        batch_key: str = "batch",
        id_key: Optional[str] = "index",
        progress: bool = True,
        return_embs: bool = False,
    ) -> Optional[torch.Tensor]:
        """Iterate over a dataloader, encoding and indexing every batch.

        Args:
            dataloader: Iterable of batch dicts.
            emb_key (str): Key for embeddings in the batch dict.
            batch_key (str): Key for the batch index tensor.
            id_key (Optional[str]): Key for vector IDs.
            progress (bool): Show a tqdm progress bar.
            return_embs (bool): If ``True``, collect and return all embeddings.

        Returns:
            Concatenated embedding tensor if ``return_embs`` is ``True``,
            otherwise ``None``.
        """
        embs = []
        for batch in tqdm(dataloader, desc="Adding to index", disable=not progress):
            emb, ids = self.add_batch(
                batch, emb_key=emb_key, batch_key=batch_key, id_key=id_key
            )
            if return_embs:
                embs.append(emb.cpu())
        if return_embs:
            embs = torch.cat(embs, dim=0)
            return embs

    def _index_database(
        self,
        batch_size: int = 2048,
        num_workers: int = 32,
        prefetch_factor: int = 4,
        id_key: str = "db_idx",
        progress: bool = True,
    ) -> None:
        """Encode every molecule in a Lance database and add it to the index.

        Builds a `LanceDataset`/dataloader over `db_file_path`, encodes each
        batch and adds the resulting embeddings to the FAISS index.
        """
        assert self.db_file_path is not None, "db_file_path must be provided."
        db_file_path = path_to_str(self.db_file_path)
        ds = LanceDataset(db_file_path)
        dl = ds.to_dataloader(
            batch_size=batch_size,
            chunked=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )
        self.add_from_dataloader(dl, id_key=id_key, progress=progress)

    def encode_smiles(self, smiles: Union[List[str], str], batch_size: int = 256) -> torch.Tensor:
        """Encode one or more SMILES strings to embeddings.

        Args:
            smiles: A single SMILES string or a list thereof.
            batch_size (int): Molecules processed per forward pass.

        Returns:
            Embedding tensor of shape ``(N, D)``.
        """
        with self.autocast_context():
            emb = self.model.encode_smiles(smiles, emb_key="emb", batch_size=batch_size)
        return emb

    def add_smiles(self, smiles: Union[List[str], str]) -> None:
        """Encode SMILES and add the resulting embeddings to the index.

        Args:
            smiles: A single SMILES string or a list thereof.
        """
        emb = self.encode_smiles(smiles)
        self.add(emb)

    def search(self, emb: Union[np.ndarray, torch.Tensor], k: int = 10):
        """Search the index for the nearest neighbours of each query embedding.

        Args:
            emb: Query embeddings of shape ``(N, D)`` as a numpy array or
                tensor.
            k (int): Number of nearest neighbours to return.

        Returns:
            Tuple ``(scores, indices)`` where ``scores[i, j]`` is the cosine
            similarity (``1 - FAISS inner-product distance``) and
            ``indices[i, j]`` is the FAISS ID of the j-th nearest neighbour
            for query i.  Both arrays have shape ``(N, k)``.
        """
        if self.shard_index_multi_gpu and isinstance(emb, torch.Tensor):
            if emb.device.type == "cuda":
                emb = emb.cpu()
        dist, index = self.index.search(emb, k)
        score = 1 - dist
        if isinstance(score, torch.Tensor):
            score = score.cpu().numpy()
            index = index.cpu().numpy()
        return score, index

    def search_extended(
        self,
        emb: Union[np.ndarray, torch.Tensor],
        k: int = 10,
        noise: float = 0.05,
        expand_times: int = 1024,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search with noise-augmented query expansion for better recall.

        Adds ``expand_times - 1`` noisy copies of each query, searches all of
        them, and deduplicates results by FAISS ID, returning the top-``k``
        unique hits by score.

        Args:
            emb: Query embeddings, shape ``(N, D)`` (tensor).
            k (int): Number of unique results to return.
            noise (float): Standard deviation of Gaussian noise added to
                query copies before L2-normalisation.
            expand_times (int): Total number of query copies (1 original +
                ``expand_times - 1`` noisy).

        Returns:
            Tuple ``(scores, indices)`` each of shape ``(1, k)``.
        """
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        emb_noised = emb.repeat(expand_times - 1, 1)
        emb_noised = torch.nn.functional.normalize(
            emb_noised + noise * torch.randn_like(emb_noised)
        )
        emb_noised = torch.cat([emb, emb_noised], dim=0)
        score, index = self.search(emb_noised, k=k)
        score = score.flatten()
        index = index.flatten()
        index, unique_idxs = np.unique(index, return_index=True)
        score = score[unique_idxs]
        sorted_indices = np.argsort(-score)
        score = score[sorted_indices][None, :]
        index = index[sorted_indices][None, :]
        return score, index

    def __call__(self, smiles: str, k: int = 10, extended: bool = False) -> pd.DataFrame:
        """Encode a SMILES string, search the index, and look up metadata.

        Args:
            smiles (str): Query SMILES string.
            k (int): Number of nearest neighbours to retrieve.
            extended (bool): Use :meth:`search_extended` for
                noise-augmented search instead of plain :meth:`search`.

        Returns:
            pandas.DataFrame with the top-``k`` database rows plus ``score``
            and ``store_index`` columns.
        """
        assert isinstance(smiles, str), "Only single SMILES string supported for __call__. Use encode_smiles + search for batch encoding/searching."
        assert self.store is not None, "store must be provided for searching by SMILES."
        emb = self.encode_smiles(smiles)
        if extended:
            scores, indices = self.search_extended(emb, k=k)
        else:
            scores, indices = self.search(emb, k=k)
        assert len(indices) == 1
        match_df = self.store[indices[0]]
        match_df["score"] = scores[0]
        match_df["store_index"] = indices[0]
        return match_df

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str,
        n_probe: int = 512,
        index_on_gpu: bool = False,
        model_on_gpu: bool = False,
        db_file_path: Optional[str] = None,
        index_file_path: Optional[str] = None,
        mmapped: bool = False,
        compile_encoder: bool = False,
        amp_dtype: Optional[str] = None,
        matmul_precision: Optional[str] = None,
        overwrite_index: bool = False,
        num_workers: int = 32,
    ):
        """Construct a :class:`SANDIndex` from a checkpoint and config file.

        Args:
            checkpoint_path (str): Path to the Lightning ``.ckpt`` file.
            config_path (str): Path to the YAML config used during training.
            n_probe (int): IVF cells to search at query time.
            index_on_gpu (bool): Move the FAISS index to GPU.
            model_on_gpu (bool): Move the encoder model to GPU.
            db_file_path (Optional[str]): Path to molecule database.
            index_file_path (Optional[str]): Path to save/load the FAISS
                index.
            mmapped (bool): Memory-map the index file.
            compile_encoder (bool): Compile the GNN encoder.
            amp_dtype (Optional[str]): AMP dtype string.
            matmul_precision (Optional[str]): Matmul precision override.
            overwrite_index (bool): Rebuild the index even if a file exists.
            num_workers (int): DataLoader workers for database indexing.

        Returns:
            Initialised :class:`SANDIndex` instance.
        """
        if not model_on_gpu:
            map_location = torch.device("cpu")
        else:
            map_location = None
        model = SANDModel.load_from_checkpoint(
            checkpoint_path, config_path, map_location=map_location
        ).eval()
        if model_on_gpu:
            model = model.to("cuda")
        return cls(
            model=model,
            n_probe=n_probe,
            index_on_gpu=index_on_gpu,
            model_on_gpu=model_on_gpu,
            db_file_path=db_file_path,
            index_file_path=index_file_path,
            mmapped=mmapped,
            compile_encoder=compile_encoder,
            amp_dtype=amp_dtype,
            matmul_precision=matmul_precision,
            overwrite_index=overwrite_index,
            num_workers=num_workers,
        )

    def save_to_disk(self, path: str) -> None:
        """Serialise the FAISS index to disk.

        Args:
            path (str): Destination file path.
        """
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), path)

    def load_from_disk(self, path: Union[str, Path], mmapped: bool = False) -> None:
        """Load a FAISS index from disk, optionally moving it to GPU.

        Args:
            path: Path to the serialised FAISS index file.
            mmapped (bool): Open the file as a memory-mapped read-only index.
        """
        path = path_to_str(path)
        if mmapped:
            index = faiss.read_index(path, faiss.IO_FLAG_MMAP)
        else:
            index = faiss.read_index(path)
        index.nprobe = self.n_probe
        if self.index_on_gpu:
            if self.shard_index_multi_gpu:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = self.f16_codes
                index = faiss.index_cpu_to_all_gpus(index, co=co)
            else:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = self.f16_codes
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index, co)
        self.index = index
        if self.store is not None and len(self.store) != self.index.ntotal:
            logger.warning(
                f"Loaded FAISS index from {path} has {self.index.ntotal} vectors, "
                f"but the database store {self.db_file_path} has {len(self.store)} rows. "
                "They may be out of sync."
            )


def merge_invlists(il_src, il_dest) -> None:
    """
    merge inverted lists from two ArrayInvertedLists
    """
    assert il_src.nlist == il_dest.nlist
    assert il_src.code_size == il_dest.code_size

    for list_no in range(il_src.nlist):

        il_dest.add_entries(
            list_no,
            il_src.list_size(list_no),
            il_src.get_ids(list_no),
            il_src.get_codes(list_no),
        )


def merge_faiss_indices(index_paths: List[str], output_path: str) -> None:
    """Merge multiple FAISS indices stored on disk into a single index."""
    assert len(index_paths) >= 2, "At least two indices are required for merging."

    # Load the first index
    merged_index = faiss.read_index(index_paths[0])

    for path in index_paths[1:]:
        index_to_merge = faiss.read_index(path)
        merge_invlists(index_to_merge.invlists, merged_index.invlists)

    faiss.write_index(merged_index, output_path)
