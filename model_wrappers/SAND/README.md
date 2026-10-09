# SAND

This is the corresponding repository for the paper *"Learning Compressed
Shape-Aware Molecular Representations for Virtual Screening"*, which proposes
**SAND** (Shape Aware Neural Descriptor), a method for learning compact
molecular representations that can retrieve shape-similar molecules from
their 2D graph alone.

<p align="center">
  <img src="model.png" alt="SAND model architecture" width="960">
</p>

If you have any questions, feel free to reach out: robin.winter{at}pfizer.com


## 1. Installation

Clone the repository and create the environment. Use `uv` for CPU or supported GPUs
(such as L40S) that can make use of PyPI faiss wheels:

```bash
# CPU-only (macOS, or Linux without a GPU) — installs faiss-cpu
uv sync --extra cpu

# GPU FAISS on supported Linux GPUs — installs faiss-gpu-cu12
uv sync --extra gpu
```

### Hopper/H100 GPU FAISS

The PyPI `faiss-gpu-cu12` wheel installed by `uv sync --extra gpu` does not
include Hopper (`sm_90`) kernels. On H100 (or newer) nodes, using
`index_on_gpu=True` with that environment fails with CUDA error 209, `no kernel
image is available for execution on the device`.

Use the Pixi environment for GPU-resident FAISS on H100 (or newer) GPUs. It
installs conda's Hopper-capable `faiss-gpu` build and keeps its CUDA stack
separate from the default uv environment:

```bash
pixi install
```

## 2. Inference

Given a trained checkpoint you can encode a large molecule database, build a
FAISS index over it, and retrieve shape-similar molecules directly from a query
SMILES.

### 2.1 Download a pretrained checkpoint

Two pretrained checkpoints - one with compression (k=256 m=32) and one without - are published as GitHub Release asset and can be downloaded by e.g.:

```bash
mkdir -p checkpoints
gh release download v1.0.0\
  --repo https://github.com/pfizer-opensource/SAND \
  --pattern "sand_k256_m32.ckpt" \
  --dir checkpoints
```
### 2.2 Build the Meta data DB (Lance store)

Here we utilize a lance database to efficiently store and retrieve meta data for the indexed compounds (SMILES, properties, etc.).
E.g. download a subset .bz2 file from Enamine (https://enamine.net/compound-collections/real-compounds/real-database-subsets) and convert it:

```bash
uv run sand-enamine-bz2-to-lance-db \
    /path/to/enamine.cxsmiles.bz2 \
    /path/to/enamine.lance
```


### 2.3 Build the index

Point `Index.from_checkpoint` at the checkpoint, its training config, the
Lance database with the SMILES to be indexed, and an output path for the FAISS index. On one H100, this indexes ~40,000 mols/s.

```python
from sand import Index

index = Index.from_checkpoint(
    checkpoint_path="/path/to/model.ckpt",
    config_path="/path/to/config.yaml", # e.g. sand/config/compression.yml
    index_file_path="/path/to/enamine.faiss",
    db_file_path="/path/to/enamine.lance",
    index_on_gpu=True,  # set False for CPU-only
    model_on_gpu=True, # set False for CPU-only
    compile_encoder=True, # set None for CPU-only
    amp_dtype="bf16",  # set None for CPU-only
    matmul_precision="high", # set None for CPU-only
    num_workers=32,
)
```
### 2.4 Search the index

Retrieve the top-k shape-similar molecules for a query SMILES

```python
match_df = index(query_sml, k=10)   # DataFrame with columns: smiles, id, score, ...
```



---

## 3. Retraining from scratch


### 3.1 Generate the training zarr stores

SAND trains on a dataset of `(query, match, score)` triples, where `score` is e.g. 3D alignment score or FP similarity.

- `graphs.zarr` — a flat CSR-style store of pre-featurised molecular graphs.
- `scores.zarr` — the `(query_idx, match_idx, score)` pairs (indices into the
  SMILES list used to build `graphs.zarr`), split into `train`/`val` groups.

Build them with the writers in `sand.data.writer`:

```python
from sand.data.writer import build_graph_store, append_score_pairs

# 1. Featurise all molecules into graphs.zarr (row order is preserved)
build_graph_store(smiles_list, "graphs.zarr", n_workers=8)

# 2. Write the precomputed score pairs (indices into `smiles_list`), per split
append_score_pairs(
    "scores.zarr", split="train",
    query_idx=query_idx_train, match_idx=match_idx_train, scores=scores_train,
)
append_score_pairs(
    "scores.zarr", split="val",
    query_idx=query_idx_val, match_idx=match_idx_val, scores=scores_val,
)
```

### 3.2 Launch training with a bundled config

Training is driven by a
[`LightningCLI`](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)
config file. Two configs are provided in `sand/config/`:

- `compression.yml` — the full SAND model **with** embedding quantization
  (IVF-PQ), M=32
- `no_compression.yml` — the same model **without** quantization (dense
  embeddings only).

Launch training:

```bash
uv run sand-train fit --config sand/config/compression.yml \
  --data.graph_storage_path=path/to/graphs.zarr \
  --data.score_storage_path=path/to/scores.zarr
```

Checkpoints and CSV logs are written under `./logs/`. Adjust `trainer.devices`,
`trainer.strategy`, `batch_size`, etc. to match your hardware.

---

## Repository layout

| Path | Contents |
|------|----------|
| `sand/nn/` | Encoder, GNN layers, quantizer, backbone |
| `sand/mol_utils/` | RDKit molecular featurization |
| `sand/data/` | Zarr graph/score store writers + `LightningDataModule` |
| `sand/index/` | FAISS-backed similarity index + on-disk molecule stores (HDF5 / Lance) |
| `sand/lit/`, `sand/output/` | Lightning training module, losses / metrics, logging |
| `sand/config/` | Example training configs (`compression.yml`, `no_compression.yml`) |
