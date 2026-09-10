# Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning

[![arXiv](https://img.shields.io/badge/arXiv-2508.06199-b31b1b.svg)](https://arxiv.org/abs/2508.06199)

---

This is a fork of the repository for "Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning" by Mateusz Praski, Jakub Adamczyk, Wojciech Czech ([paper](https://arxiv.org/abs/2508.06199)), adapted by Inductive Bio in two ways:

- **Datasets**: the original TDC/MoleculeNet benchmarks are replaced with 17 regression endpoints from three OpenADMET challenges — [ExpansionRx](docs/data_sources/expansionrx.md), [Polaris antiviral ADMET/Potency](docs/data_sources/polaris_antiviral.md), and [PXR](docs/data_sources/pxr.md) — all of which concluded and were unblinded after every embedding model's pretraining cutoff. The original TDC/MoleculeNet tasks predate that cutoff, so a top-performing model there may simply have seen the test molecules during its own pretraining.
- **Ranking methodology**: embeddings are ranked with a Bayesian latent-ability model that directly models the margin of error between models (the difference in log MAE) rather than reducing each comparison to a win/loss, as the original Bayesian Bradley-Terry approach does. That reduction discards information and understates ranking uncertainty (proven in [docs/bayesian_model.md](docs/bayesian_model.md); see "Ranking embeddings" below).

## Requirements

- Python >= 3.10, < 3.12
- Pyenv for managing embedding model venvs

## Installation

After you create a fresh venv (Python >=3.10,<3.12), run the following command to install the dependencies:

```sh
./install_deps.sh
```

## Usage

### 1. Prepare the OpenADMET datasets

Download and clean the raw challenge data into `data/processed/<source>/<endpoint>.csv`:
```sh
python scripts/prepare_openadmet_data.py
```
Then materialize the resulting `config/dataset/reg_openadmet_*.yaml` datasets, same as any other dataset:
```sh
python download.py --multirun '+dataset=glob(reg_openadmet_*)'
```

### 2. Compute embeddings

To run the embedding procedure for an existing model, for all datasets, run the command:
```sh
./embed_wrapper.sh /path/to/implementation # e.g. ./embed_wrapper.sh model_wrappers/huggingface
```

If you want to run this in the background, you can use the shorthand command:
```sh
./run_embed.sh <implementation name> # e.g. ./run_embed.sh huggingface
```

This will perform the embedding procedure on all defined datasets (see `config/embed.yaml` for the default list). `scripts/modal_embed.py` runs the same procedure on Modal, one container image per wrapper, for wrappers that don't need a legacy Python interpreter.

List of available implementations is defined under `config/experiment`.
List of available datasets is defined under `config/dataset`.

Embeddings will be saved as an [EmbeddedDataset](src/common/types.py) object under `data/embedded/<dataset>/model.<joblib/json>`, depending on the implementation. Due to the Pandas serialization format incompatibility in different environments, we implement a simple custom serialization of outputs from some older models.

In our benchmark study, some of the models required filtering of a small number of SMILES, e.g.: due to the SELFIES translation issues, or lack of support for dative bonds. Those SMILES are defined in [illegal_smiles.txt](config/illegal_smiles.txt) in the config directory. To reproduce those illegal SMILES as in the original articles, please refer to the `selfies_debugger.ipynb` notebook.

### 3. (optional) Fine-tune end-to-end

Several wrappers also support end-to-end fine-tuning of the underlying pretrained model, instead of only scoring its frozen embeddings, via `model_wrappers/<wrapper>/finetune.py`. `scripts/modal_finetune_<wrapper>.py` runs the equivalent procedure on Modal.

### 4. Compute per-compound prediction errors

For each (embedding, regression head) pair, this fits on an endpoint's train+valid split (hyperparameters chosen by CV) and stores the per-compound test-set prediction error:
```sh
python scripts/compute_prediction_errors.py --embedding ECFP --method rf
```
Supported heads: `rf`, `ridge`, `knn`, `tabicl`, `tabpfn`. `scripts/modal_score_errors.py` fans this out across Modal containers, one per (embedding, method, endpoint).

### 5. Rank embeddings with the Bayesian latent-ability model (BLAM)

`scripts/blam_analysis.py` fits the hierarchical model described in [docs/bayesian_model.md](docs/bayesian_model.md) via NUTS (numpyro): for each endpoint, it picks each embedding's best-by-CV regression head, builds the per-compound error matrix, and fits a shared latent ability `θ` per embedding alongside per-endpoint difficulty/discrimination/specialization terms. Unlike a Bradley-Terry-style ranking, this models the population log-MAE margin between models directly, so uncertainty in the ranking isn't inflated by treating derived pairwise wins as independent observations.
```sh
python scripts/blam_analysis.py --errors-dir data/errors --output data/blam
```
`scripts/modal_blam_analysis.py` runs the fit on Modal. `scripts/visualize_blam.py` turns the posterior into ranking charts and interactive dashboards. `scripts/blam_robustness_checks.py` and `scripts/pl_variance_inflation.py` (the latter a Monte Carlo check of the variance-inflation derivation itself) validate the model's assumptions.

## Testing your own model

To test your own model, please implement a wrapper following the [instructions here](docs/custom_model.md). You can also refer to the existing implementations under `model_wrappers` directory.

### Sharing your model

If you want your model to be included in our latest benchmark, feel free to open a PR with your implementation. We will be happy to review it and include it in the results.

## Tips on controlling number of threads

This project relies on hydra multirun functionality to run experiments in parallel. By default, the embedding procedure does not perform multirun, however if you want to run multiple datasets at once, you can specify parameter `hydra.launcher.n_jobs` in the experiment config (`config/experiment/model_name.yaml`, see [fingerprints](config/experiment/fingerprints.yaml) for an example). When using multirun mode, you also need to override hydra launcher from default to `joblib` (as in fingerprints example). See more about hydra multirun [here](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) and [here](https://hydra.cc/docs/plugins/joblib_launcher/).

## Cite 

If you find this work useful in your research, please cite:

```
@article{praski2025benchmarking,
  title={Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning},
  author={Praski, Mateusz and Adamczyk, Jakub and Czech, Wojciech},
  journal={arXiv preprint arXiv:2508.06199},
  year={2025}
}
```

## License

Feel free to use any parts of this project in your research. If you do so, please cite our paper as shown above. Due to the vast number of models used in this project, we're unable to provide a unified license for the entire repository. Please refer to the individual model implementations for their respective licenses.
