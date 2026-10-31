# A Latent Ability Model for Benchmarking Molecular Representations

This repository contains the code and analysis scripts for "A Latent Ability Model for Benchmarking Molecular Representations". It extracts embeddings from pretrained chemical models, fine-tunes a subset of them end to end, scores everything on 17 ADMET regression endpoints, and ranks the resulting 48 representations (35 frozen embeddings and 13 end-to-end fine-tuned models) with a Bayesian latent ability model.

The ranking model directly models each model's per-endpoint MAE as a function of latent ability, rather than reducing every comparison to a pairwise win as the Bayesian Bradley-Terry (BBT) model does. Under the exchangeable null, the BBT pairwise pseudo-likelihood inflates the variance of its estimate by a factor of (M+1)/3 for M models, independent of the number of endpoints. The latent ability model has one likelihood factor per endpoint, so it does not have this problem. See Sec. 3 and Apps. A and B of the paper for the derivation and model details. A standalone implementation of the method for any set of regression tasks scored on shared test sets is available at [nedassan/blam-analysis](https://github.com/nedassan/blam-analysis).

## Datasets

The 17 endpoints come from three OpenADMET blind challenges: [ExpansionRx](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data) (9 endpoints), Polaris antiviral ADMET and potency (7 endpoints), and [PXR](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test) (1 endpoint). Each was measured under a single protocol and released with a held-out test set, which is used as-is. All endpoints are modeled on a log10 scale. See [docs/datasets.md](docs/datasets.md) for sources, transforms, and exclusions.

## Requirements

- Python >= 3.10, < 3.12
- pyenv, for the per-model virtual environments

## Installation

In a fresh virtual environment, install the base dependencies:

```sh
./install_deps.sh
```

Each model wrapper under `model_wrappers/` gets its own virtual environment, created by `embed_wrapper.sh` from that wrapper's `.python-version` and `init.sh`.

## Usage

### 1. Prepare the datasets

Clone the Polaris challenge examples repository, which holds the organizers' exclusion lists, then download and clean all 17 endpoints into `data/processed/<source>/<endpoint>.csv`:

```sh
git clone https://github.com/asapdiscovery/asap-polaris-blind-challenge-examples data_repos/asap-polaris-blind-challenge-examples
python scripts/prepare_openadmet_data.py
python download.py --multirun '+dataset=glob(reg_openadmet_*)'
```

### 2. Compute embeddings

To embed every dataset with one wrapper:

```sh
./embed_wrapper.sh model_wrappers/huggingface
```

`./run_embed.sh <wrapper>` runs the same thing in the background. The datasets are listed in `config/embed.yaml`, the available wrappers and their model variants in `config/experiment/`. Embeddings are saved as an [EmbeddedDataset](src/common/types.py) under `data/embedded/<dataset>/<model>.<joblib|json>`. Some older wrappers use a custom JSON serialization because their environments have incompatible pandas versions.

`scripts/modal_embed*.py` run the same procedure on [Modal](https://modal.com). Descriptors computed on 3D conformers (Mordred with 3D descriptors and RDKit-3D, averaged over each molecule's conformer ensemble with Boltzmann or uniform weights) are produced by `scripts/modal_generate_conformers.py`, `scripts/modal_optimize_conformers.py`, `scripts/modal_embed_descriptors.py`, and `scripts/modal_embed_aggregated_conformers.py`.

A small number of SMILES that some models cannot process (for example SELFIES translation failures or dative bonds) are listed in [config/illegal_smiles.txt](config/illegal_smiles.txt) and filtered out before embedding.

### 3. Fine-tune end to end

Models whose authors provide an end-to-end regression fine-tuning method are fine-tuned on each endpoint via `model_wrappers/<wrapper>/finetune.py` (`finetune_selformer.py` for SELFormer), following that method with one default learning rate per model (no per-endpoint search). Chemprop (2D and 3D features) is trained end to end from scratch via `model_wrappers/chemprop/finetune.py`. Fine-tuning scripts for SimSon and GROVER are also included but were not part of the paper's results. Results are written to `data/finetune_errors/<dataset>/<model>__finetuned.joblib`. `scripts/modal_finetune_<wrapper>.py` runs the same procedure on Modal.

### 4. Score frozen embeddings

For each (embedding, regression head) pair, `scripts/compute_prediction_errors.py` fits the head on the endpoint's training split and stores the per-compound test-set error:

```sh
python scripts/compute_prediction_errors.py --embedding ECFP_count --method rf
```

Supported heads, with hyperparameters chosen by 5-fold CV (App. C.2). The CV score is Spearman correlation (the `metric` in each `config/dataset/*.yaml`), and it is also what `blam_analysis.py` uses to pick each embedding's best head:

| Head | Search |
|---|---|
| `rf` | `min_samples_split` in {2, 4, 6, 8, 10}; 200 trees, `max_depth=20` |
| `ridge` | `alpha` in 10 log-spaced values over [1e-2, 1e3], standardized features |
| `knn` | `n_neighbors` in {1, 3, 5, 7, 9}, standardized features; Tanimoto-count distance for count embeddings, cosine otherwise |
| `tabicl` | no search; 5-fold CV score, then one fit on all training data |
| `tabpfn` | no search (TabPFN v3); same as `tabicl` |

TabICL and TabPFN inputs are standardized and clamped to [-6, 6]. TabPFN's default checkpoint is gated on Hugging Face, so it needs an `HF_TOKEN` with access granted.

`scripts/modal_score_errors.py` fans this out on Modal, one container per (embedding, head, endpoint). It runs only `rf`, `ridge`, and `knn` by default. `blam_analysis.py` only includes an embedding on an endpoint once all five heads have results, so pass `--methods rf,ridge,knn,tabicl,tabpfn`.

### 5. Fit the latent ability model

`scripts/blam_analysis.py` selects each embedding's best head per endpoint by CV score, builds the per-compound error matrix, and fits the model with NUTS in NumPyro (16 chains, 2,000 warmup and 4,000 sampling steps by default). To reproduce the 48 representations in the paper's Table 2:

```sh
python scripts/blam_analysis.py \
    --errors-dir data/errors \
    --finetune-errors-dir data/finetune_errors \
    --embeddings-file config/blam/paper_embeddings.txt \
    --finetuned-models-file config/blam/paper_finetuned_models.txt \
    --output data/blam
```

The two list files hold the 35 frozen embeddings and 13 fine-tuned models used in the paper. Frozen model families with several variants were pruned to the variant with the best CV score; each fine-tuned family was run with a single variant. The script warns about any listed model that has no results, and writes `posterior.joblib` and summary CSVs for latent ability, per-endpoint parameters, and per-endpoint specialization. `scripts/modal_blam_analysis.py` runs the chains as separate Modal containers.

### 6. Reports and checks

- `scripts/visualize_blam.py` turns the posterior into ranking charts, comparisons against the ECFP (count) baseline on observed and new endpoints (App. C.5, including Table 7's P(δ > 0), 90% interval, and new-endpoint win rate via `--compare-ecfp`), and interactive dashboards. Pass `--display-names config/blam/model_display_names.json` to use the paper's model labels.
- `scripts/pl_variance_inflation.py` (and `scripts/modal_pl_variance_inflation.py` for the full NUTS replicate count) simulates BBT on Plackett-Luce rankings to check the variance inflation factor (App. A.6).
- `scripts/blam_robustness_checks.py` runs additional sensitivity checks on the fitted model in `data/blam` (not reported in the paper).

## Adding a model

To benchmark another model, implement a wrapper following [docs/custom_model.md](docs/custom_model.md); the existing wrappers under `model_wrappers/` are working examples.

## License

Model wrappers include or download third-party code and weights, each under its own license. See the individual directories under `model_wrappers/` for details.
