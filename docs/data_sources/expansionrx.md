# OpenADMET ExpansionRx Challenge — Data Guide

> **Status:** downloaded and cleaned by `scripts/prepare_openadmet_data.py`. Raw pulls
> in `data/raw/expansionrx/`, 9 per-endpoint cleaned+log-transformed CSVs in
> `data/processed/expansionrx/`. Loaded straight from the existing local HF cache, no
> network/auth issues. `LogD` kept raw; the other 8 endpoints got the
> `log10((x+1)*multiplier)` transform per §4 below.

Source tutorial reviewed: `data_repos/ExpansionRx-Challenge-Tutorial` (`expansion_tutorial.ipynb`, `eval/`).

This challenge has already concluded (submission deadline was 2026-01-19), and a fully
labeled version of the dataset (train + test, both with real assay values) is publicly
available — we do **not** need to go through the live blind-submission workflow the
tutorial notebook demonstrates.

## 1. Data source

Use the Hugging Face dataset repo **`openadmet/openadmet-expansionrx-challenge-data`**
(not `...-train-data` and not `...-test-data-blinded` — those only contain the train
split or an unlabeled test split, respectively). This repo has both splits with real
labels:

```python
import pandas as pd

train_df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-data/expansion_data_train.csv")
test_df  = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-data/expansion_data_test.csv")
```

Confirmed locally: this repo is already cached at
`~/.cache/huggingface/datasets/openadmet___openadmet-expansionrx-challenge-data/` — 5326
train rows, 2282 test rows, and another repo in this org
(`~/code/ensemble-aware-admet/conformer_generation/expansionrx.py`) already loads the
test split from this exact repo (`split="test"`) as ground truth, confirming it's the
correct source to use. No authentication required (public dataset).

Requires `pandas` with `fsspec`/`huggingface_hub` installed for the `hf://` path to
resolve (already fine per the tutorial's `environment.yaml`).

## 2. Schema

Both splits share the same columns:

| Column | Type | Notes |
|---|---|---|
| `Molecule Name` | str | unique compound ID, e.g. `E-0001321` |
| `SMILES` | str | plain SMILES (not CXSMILES) |
| `LogD` | float | already log-scale |
| `KSOL` | float | kinetic solubility, µM |
| `HLM CLint` | float | human liver microsomal clearance |
| `MLM CLint` | float | mouse liver microsomal clearance |
| `Caco-2 Permeability Papp A>B` | float | ×10⁻⁶ cm/s |
| `Caco-2 Permeability Efflux` | float | efflux ratio |
| `MPPB` | float | mouse plasma protein binding |
| `MBPB` | float | mouse brain protein binding |
| `MGMB` | float | mouse gastrocnemius muscle binding |

This is **9 separate regression endpoints**, each sparse (most rows have only a handful
of the 9 columns populated — this is a real drug-discovery program's assay cascade, not
every compound was run through every assay). Treat each endpoint as its own single-task
regression dataset, mirroring how this benchmark already treats each TDC ADMET assay as
one `config/dataset/*.yaml` entry. That means **9 dataset configs** for this challenge
(e.g. `reg_expansionrx_LogD.yaml`, `reg_expansionrx_KSOL.yaml`, ...).

## 3. Cleaning / filtering steps

For each endpoint you're building a dataset for:

1. Select `["Molecule Name", "SMILES", <endpoint column>]` from the combined
   train+test frame.
2. Drop rows where the endpoint column is `NaN` — sparsity means each endpoint ends up
   with a different, smaller set of rows than the full 5326/2282 train/test sizes.
3. Canonicalize SMILES the same way the rest of this repo does
   (`Chem.CanonSmiles`, done centrally in `build_dataset()` in `src/common/data_v2.py`
   — no need to duplicate that here).
4. Rename `Molecule Name` → whatever id-column convention `Dataset.labels` expects
   (anything with `id` in the name gets dropped from labels automatically — see
   `src/common/types.py`).
5. Keep the existing train vs. test row membership from the two source files (don't
   re-split) — this preserves whatever train/test partitioning OpenADMET originally
   used (their tutorial doesn't state whether it's scaffold/time/random split; that
   detail wasn't in the notebook and would need to come from OpenADMET directly if it
   matters for leakage analysis).

## 4. Log transform (open decision, not mandatory)

The tutorial log-transforms 8 of the 9 endpoints before modeling (`LogD` is already
log-scale), using `log10((x + 1) * multiplier)` with these multipliers:

| Assay | Log? | Multiplier |
|---|---|---|
| LogD | no | — |
| KSOL | yes | 1e-6 |
| HLM CLint | yes | 1 |
| MLM CLint | yes | 1 |
| Caco-2 Papp A>B | yes | 1e-6 |
| Caco-2 Efflux | yes | 1 |
| MPPB | yes | 1 |
| MBPB | yes | 1 |
| MGMB | yes | 1 |

This is a modeling convenience (normalizes skew, stabilizes variance for the linear/tree
baselines they use), not part of the raw data itself. Recommend storing raw values in
`data/prepared/*.joblib` and deciding at eval-metric time whether MAE/Spearman should be
computed in log space or raw space — flag this as a decision for whoever's re-doing the
statistical-testing pass, since it changes what "MAE" means per endpoint.

## 5. Integration point in this repo

- New `source.name` (e.g. `OPENADMET_EXPANSIONRX`) branch in `src/common/data_v2.py::load()`.
- Solver reads the HF CSVs above, returns `(DataFrame, splits)` where `splits` is built
  from the existing train/test row membership (no `valid` split provided by the source —
  same situation as the current `OGB` solver, which also leaves `valid` empty).
- 9 new `config/dataset/reg_expansionrx_*.yaml` files, `task: regression`.
- No graph-column support needed unless you want ExpansionRx to also feed
  GraphEmbedder-based models — replicate the `smiles2graph` + `MolConvert` step from the
  existing TDC/OGB solvers if so.
