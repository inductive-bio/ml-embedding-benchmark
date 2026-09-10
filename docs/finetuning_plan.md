# Fine-tuning plan

## Context

The benchmark currently scores pretrained molecular embedding models by freezing each model's encoder, extracting a fixed-length embedding per molecule (`src/embedding/embedding.py`), and fitting a classical sklearn head — RF/Ridge/KNN/TabICL (`src/eval/supervised/models.py`, `regression.py`) — on top per ADMET endpoint. After the OpenADMET migration (`docs/openadmet_migration_status.md`), this is scored across 17 post-cutoff regression endpoints (ExpansionRx, Polaris antiviral ADMET/Potency, PXR), with embeddings for every wrapper cached in `data/embedded/` and per-(dataset, embedding, head) errors in `data/errors/`, feeding a Bayesian ranking model (`docs/bayesian_model.md`, `data/blam/`).

Looking at `data/blam/endpoint_summary.csv`/`theta_summary.csv`, absolute error is mediocre under this architecture (e.g. ~0.68 LogD units, ~0.6-0.7 log-units on potency/PXR endpoints), and the top-ranked "embeddings" by latent ability are the cheap RDKit/Mordred descriptor baselines added in `docs/embedding_expansion_plan.md`, not the expensive learned encoders. That plan already flags the frozen-embedding + shallow-head architecture as the likely bottleneck, and explicitly defers Chemprop-style end-to-end training as not fitting it.

Several of the wrapped models were originally released with their own fine-tuning code — training the encoder (or most of it) end-to-end on a downstream task rather than treating it as a frozen feature extractor. This plan identifies which wrappers actually support that (confirmed by reading each one's vendored README/code directly, not assumed from the model's reputation), and lays out how to fine-tune each on the same 17 OpenADMET endpoints, so the results can be compared against the existing frozen-embedding numbers using the same splits and metrics.

**Explicitly out of scope**: `model_wrappers/chemeleon/` and any chemprop-family model. The user has already trained chemprop + Chemeleon-style models end-to-end in the sibling repo `../ensemble-aware-admet`; bringing those model families to this repo's 17 endpoints is planned separately in `docs/e2e_chemprop_chemeleon_plan.md`.

This doc is a plan, not an implementation — no fine-tuning code or runs exist yet.

## Scope: which wrappers actually support fine-tuning

Confirmed per-wrapper by reading each one's vendored README/code.

### Tier 1 — ready-made fine-tuning CLI, modern Python env, GPU already wired up via Modal

| Wrapper | Fine-tune entry point | Notes |
|---|---|---|
| `unimol` | `unimol_tools.train.MolTrain` (ships in the `unimol_tools` pip package; current `wrapper.py` only ever calls the frozen `UniMolRepr.get_repr()` path) | Python ≥3.10, already Modal-GPU-capable (`scripts/modal_embed_unimol.py`); the only genuinely 3D-at-inference model in the benchmark. `finetune.py --model-name` already supports both `unimolv1`/`unimolv2` and `modal_finetune_unimol.py` already fans out over both — confirmed live that `unimolv2`'s pair-representation ("outer product mean") layers need far more memory per molecule than v1's plain per-atom transformer: the shared default `batch_size=16` OOMs on both a T4 (16GB) and an A10G (24GB) in the same layer, and only an A100 (40GB, via the existing `--modal-gpu` override, no code change) had headroom |
| `gem` | `gem_repo/apps/pretrained_compound/ChemRL/GEM/scripts/finetune_regr.sh` | Already Modal-GPU-capable for embedding; README documents ~1-2 days on a single V100 for a full MoleculeNet-scale finetuning sweep — our per-endpoint runs should be much smaller |
| `huggingface` (SELFormer variant) | `selformer_repo/train_regression_model.py` — full `RobertaForSequenceClassification` fine-tune (encoder + head), not a linear probe | Already has a dedicated Modal script (`scripts/modal_embed_selformer.py`) to extend |
| `huggingmolecules` (MAT / R-MAT / GROVER) | Manual PyTorch training loop reusing `MatModel`/`RMatModel`/`GroverModel` + their featurizers directly (not `experiments/src/training`'s PyTorch-Lightning `TrainingModule` — that module's own metrics import `pytorch_lightning.metrics`, an API removed after PL~1.3, and `experiments/requirements.txt` pins `pytorch_lightning==1.0.6`, incompatible with the torch==2.8 this venv already needs for every other wrapper) | Already running on Modal T4 for embedding; lr/batch_size/epochs defaults read directly off `experiments/configs/models/{mat,rmat,grover}.gin` |

Chemformer moved to Tier 3 below — confirmed by reading the vendored repo that, despite having a `fine_tune.py` entry point, it has no scalar-regression fine-tuning path at all (see that entry).

### Tier 2 — fine-tuning documented, but legacy Python (3.7-3.9)

Originally scoped as "CPU-only (Modal can't build cp37 images)" — turned out only half true. Modal's own client library does require the container's primary interpreter to be >=3.10, but nothing stops a *second*, fully separate legacy interpreter from living side by side in the same image, used only via subprocess — confirmed working end-to-end for `molbert` (see `model_wrappers/molbert/Dockerfile.modal`): copy the interpreter tree straight out of the official `python:<version>-bullseye` image (~72s image build, vs. ~28 minutes to compile the same Python from source — confirmed empirically) rather than trying to bump the wrapper's own pinned deps to a cp310+ wheel that doesn't exist. `torch==1.4.0`'s PyPI wheel turned out to bundle working CUDA support too, confirmed on a Modal T4 — so Tier 2 isn't even CPU-only-by-necessity on Modal, just per-wrapper-dependent (see `graphfp` below, whose own pin *is* CPU-only by design).

| Wrapper | Fine-tune entry point | Notes |
|---|---|---|
| `molbert` | `molbert/apps/finetune.py::FinetuneSmilesMolbertApp` (`--mode regression --output_size 1`, loads from a pretrained `.ckpt`) | pyenv 3.7.17; local venv CPU-only, but also has a working, **piloted end-to-end on a Modal T4** GPU pipeline now (`model_wrappers/molbert/Dockerfile.modal` + `model_wrappers/molbert/finetune.py` + `scripts/modal_finetune_molbert.py`): `reg_openadmet_pxr_pEC50`, 5 epochs, MAE 0.527 / Spearman 0.699 on 513 test molecules — competitive with the Tier 1 results above. Two real bugs found and fixed along the way: (1) `data/prepared/*.joblib` is pickled with this repo's pandas 2.3.3, but pandas dropped cp37 support after 1.3.5 — an old pandas can't unpickle a 2.x DataFrame's BlockManager at all (confirmed live). Fixed once in the shared harness (`src/common/finetune_data.py::load_finetune_splits` now caches the derived split to plain CSV, which any pandas version can read) rather than per-wrapper, since this affects every Tier 2 adapter, not just this one. (2) `BertFinetuneSmilesDataset.get_sample()`'s own over-length check compares the *character* length of a standardised SMILES against `max_length`, not the actual token count `convert_example_to_features()` will produce — a short-looking SMILES can still tokenize past `max_seq_length=128` and crash on a hard `assert` with no truncation fallback (confirmed live against a real OpenADMET row, a gold-phosphine complex). `model_wrappers/molbert/finetune.py::_drop_unencodable()` pre-filters using the real tokenized length before any split is written to CSV, rather than patching the vendored dataset class. |
| `graphmvp` (+ its 8 bundled baseline strategies: AttrMask, ContextPred, GraphCL, EdgePred, GPT-GNN, Motif, Infomax) | `scripts_regression/submit_fine_tuning.sh` | pyenv 3.7.17 local venv, CPU; small GIN |

### Tier 3 — needs custom adaptation work beyond invoking an existing script

| Wrapper | Situation |
|---|---|
| `simson` | `main.py --task downstream --criterion mse` exists and supports regression, but is less thoroughly documented than Tier 1/2 entries — needs a validation pass before committing to it |
| `SAND` | `SANDModel` is a full LightningModule with a working training loop (`sand-train fit --config ...`), but it's wired for contrastive (query, match, score) triples, not scalar ADMET regression — would need a regression head bolted onto the encoder and a new training script, not just a CLI invocation |
| `clamp` | `clamp/train.py` (train-from-scratch) + `linear_probe.py` exist, but the model is CLIP-style (molecule ↔ assay-text contrastive), which isn't a natural fit for fine-tuning against a single scalar target — needs a design decision on how to adapt it |
| `mol-r` | `main.py` has downstream-task code paths, but nothing in the vendored repo frames it as "fine-tune the pretrained encoder on a new property" the way the Tier 1/2 READMEs do |
| `chemformer` | Confirmed by reading `molbart/models/chemformer.py`, `transformer_models.py`, and `molbart/data/` in full: the vendored repo only ships `BARTModel`/`UnifiedModel`, both seq2seq (reaction-prediction/mol-opt) `LightningModule`s — no regression/property-prediction head or scalar-label dataset exists anywhere in it, despite `fine_tune.py` looking like a ready-made entry point and the README claiming the original paper tested "molecular property prediction" (that code isn't in this vendored subset). **Effort estimate for later**: would need a new `nn.Linear` regression head bolted onto `Chemformer.embed()`'s existing pooled-encoder-output (`model_output` mean-pooled over tokens, already used by `wrapper.py` today) plus a from-scratch training loop (optimizer, early stopping, checkpointing) — no existing script to invoke or PyTorch-Lightning module to wrap, unlike every other Tier 1/2 entry. Comparable in scope to `SAND`/`clamp` above, i.e. a half-to-one day of new code rather than an adapter file, plus an open question of whether to fine-tune the encoder end-to-end or freeze it and only train the new head (worth trying both, given ADMET endpoints here only have a few hundred rows).
| `graphfp` | Moved here after reading `mol/README.MD`'s "Finetuning" section and the two scripts it actually points to (`downstream_old.py`, `downstream_frag_old.py`) in full — both hardcode `nn.BCEWithLogitsLoss` and `roc_auc_score` as the only loss/metric, using the `y ∈ {-1, 0, +1}` masked-multi-task-classification convention standard for OGB/MoleculeNet benchmarks (`downstream_frag_old.py` even has a dead, commented-out `# criterion = nn.MSELoss()` line, suggesting regression was considered once but never wired in). Data loading (`loader.MoleculeDataset`, a PyG `InMemoryDataset`) expects a pre-built `chem_dataset/<name>/processed/geometric_data_processed.pt` cache, not an arbitrary SMILES+float CSV — no scalar-regression dataset class exists anywhere in `graphfp_repo/mol/`. **Effort estimate for later**: the `GNN_graphpred`/`GIN` model classes themselves are architecture-generic (a plain `Linear(emb_dim, num_tasks)` head, nothing classification-specific baked in), so a regression adapter is buildable by reusing those classes plus a new training loop — comparable in scope to the `chemformer` estimate above, not a Tier 2 CLI wire-up. Checkpoints are already vendored, no download needed (`model_wrappers/graphfp/graphfp_repo/mol/pretrain/{GIN_C.pth, GIN_CP_03.pth, GIN_CPF_01.pth}`). Separately, `graphfp`'s own pin is `torch==1.9.0+cpu` — CPU-only by design (matching `+cpu`-tagged `torch_scatter`/`torch_sparse` wheels from `data.pyg.org`), so even the eventual regression adapter would have no GPU path regardless of Modal deployability.

### Deprioritized

- **`coati`** — the wrapper hardcodes `freeze=True` when loading the encoder, and the vendored repo only ships a from-scratch pretraining script (`examples/training/train_grande.py`), not a per-task fine-tune entry point. Supporting this would mean writing a training loop from scratch rather than adapting an existing one — lowest priority of everything considered here.

### Excluded — no gradient-based fine-tuning path at all

`cddd` (seq2seq autoencoder latent, no downstream fine-tune code), `fingerprints` (hashed, non-learned), `descriptors` (hand-crafted RDKit/Mordred features, non-learned), `mol2vec` (word2vec over substructures, not a per-task gradient-trainable model).

### Excluded per user instruction

`chemeleon` and any chemprop-family model — the underlying models were built in `../ensemble-aware-admet`; bringing them to this repo's 17 endpoints is planned in `docs/e2e_chemprop_chemeleon_plan.md`.

## Common fine-tuning harness design

All included models should share as much of this as possible, so results stay comparable and the work isn't reinvented per wrapper.

**Data.** Reuse `data/processed/<source>/<endpoint>.csv` (`molecule_id, smiles, value, split`) — the same cleaned per-endpoint data the frozen-embedding pipeline already uses (`scripts/prepare_openadmet_data.py`, `src/common/data_v2.py::local_csv_solver`). That loader currently returns only `train`/`test` splits (`valid: []`). Gradient-trained models need a validation split for early stopping/model selection, unlike the sklearn heads' `GridSearchCV` internal CV — carve a fixed validation split out of `train` once (e.g. 80/20, fixed seed) and reuse the exact same split across every model, so results stay comparable and no model gets a lucky/unlucky validation carve-out.

**Per-wrapper adapter.** Each Tier 1/2 wrapper needs a small adapter (analogous to its existing `wrapper.py`) that reformats the CSV above into whatever schema/CLI its own fine-tuning script expects, then either shells out to that script or calls its training entry point directly in-process (e.g. `unimol_tools.train.MolTrain` is a plain Python class, no subprocess needed).

**Output and comparability.** Run inference on the held-out `test` split, score with the same metric functions the frozen-embedding path uses (`src/eval/supervised/utils.py::get_sklearn_scorer` — MAE, Spearman), and write results into a new parallel path, e.g. `data/finetune_errors/<dataset>/<model>.joblib`, matching the schema `scripts/compute_prediction_errors.py` already writes to `data/errors/` (`molecule_id, y_true, y_pred, error, ...`). Matching the schema means `scripts/blam_analysis.py`'s Bayesian ranking model can treat "fine-tuned end-to-end" as just another method per dataset with no changes to the ranking code itself.

**Execution environment.** Tier 1: new `scripts/modal_finetune_<model>.py` scripts, one Modal GPU container per (dataset, model) training run, mirroring the existing `scripts/modal_embed_unimol.py` / `modal_embed_selformer.py` pattern (a training run is heavier than a single embedding forward pass, so this is a bigger ask of Modal than the existing embedding jobs — validate cost/duration on one endpoint before fanning out to all 17). Tier 2: run locally via the pyenv legacy-Python recipe already documented in `docs/openadmet_migration_status.md`'s "Local run recipe" section — these are CPU-only regardless, since Modal's client library requires the container's own interpreter to be ≥3.10.

**Hyperparameters.** Start from each README's own documented defaults (LR, epochs, batch size) rather than running a full HPO sweep per endpoint — that's already an enormous compute multiplier (models × endpoints) before adding hyperparameter search on top. Flag the overfitting risk explicitly: several OpenADMET endpoints have only a few hundred rows, small relative to these pretrained encoders' capacity, so early stopping on the validation split and conservative epoch counts should be the default, with exceptions only where a wrapper's own README recommends something specific for small-data regression (e.g. SELFormer's `train_regression_model.py` already exposes `--lr`/`--num_epochs`/`--wd` directly for this).

## Tier 1 file layout

Concrete files added for the 4 Tier 1 wrappers (chemformer moved to Tier 3, see above), following the repo's existing convention of one file per concern inside each `model_wrappers/<name>/` directory. Every `finetune.py` below shares the same CLI shape — `--dataset <config_name> --output <path> [--lr ...] [--epochs ...]` — so the Modal scripts and any local runner can invoke all four uniformly.

### Shared infrastructure (new, written once, used by every wrapper)

- **`src/common/finetune_data.py`** — `load_finetune_splits(dataset_config) -> (train_df, val_df, test_df)`. Reads the same `data/processed/<source>/<endpoint>.csv` the frozen-embedding path already reads, carves the fixed validation split out of `train` (fixed seed, e.g. 80/20), and returns three DataFrames with consistent columns (`molecule_id`, `smiles`, `value`). Every per-wrapper `finetune.py` calls this first, then reshapes the result into whatever schema its own training script expects — this is the one piece that must be identical across all models for the comparison to be meaningful.
- **`src/common/finetune_results.py`** — `save_finetune_result(dataset_name, model_name, test_df_with_preds)`. Computes MAE/Spearman via the same functions `src/eval/supervised/utils.py::get_sklearn_scorer` already provides, and writes `data/finetune_errors/<dataset>/<model>.joblib` in the same `{molecule_id, y_true, y_pred, error}` schema `scripts/compute_prediction_errors.py` writes to `data/errors/`. Every `finetune.py` calls this last.

### `model_wrappers/unimol/finetune.py`

`unimol_tools.train.MolTrain` is a plain Python class — no subprocess needed. This file: calls `load_finetune_splits`, writes train+val into the DataFrame shape `MolTrain` expects (SMILES column + target column), calls `MolTrain(task='regression', ...).fit(...)`, then loads the resulting checkpoint via `unimol_tools`'s prediction class to run inference on the test split, and calls `save_finetune_result`. No new shell scripts — reuses `model_wrappers/unimol/init.sh` as-is since the environment doesn't change.

### `model_wrappers/gem/finetune.py` (+ small config generation)

GEM's `finetune_regr.sh` expects a dataset registered in its own MoleculeNet-style loader, not an arbitrary CSV path. This file: calls `load_finetune_splits`, writes train/val/test into GEM's expected on-disk layout, generates a minimal per-dataset config (task type, split file paths — reusing one of `gem_repo`'s existing regression config templates as a base rather than hand-writing 17), then invokes GEM's underlying finetune script as a subprocess pointed at that config, and finally reads GEM's own output predictions (or runs its predict path directly) and calls `save_finetune_result`.

### `model_wrappers/huggingface/finetune_selformer.py`

Separate from a generic `huggingface` fine-tune file, mirroring how `modal_embed_selformer.py` is already split out from `modal_embed.py` — SELFormer is the only `huggingface`-bundled model with a documented fine-tune path. This file: calls `load_finetune_splits`, writes train/val CSVs in the format `selformer_repo/train_regression_model.py` expects, invokes it as a subprocess (`--lr`, `--num_epochs`, `--train_batch_size`, pointing at the pretrained checkpoint), then invokes the matching `regression_pred.py` on the test split, and calls `save_finetune_result`.

### `model_wrappers/huggingmolecules/finetune.py`

Turned out not to go through PyTorch Lightning after all: `experiments/src/training`'s own `TrainingModule` depends on `experiments/src/training/training_metrics.py`, which imports `pytorch_lightning.metrics` (an API removed after PL~1.3 in favor of torchmetrics) and `experiments/requirements.txt` pins `pytorch_lightning==1.0.6` — not realistically installable alongside the torch==2.8 this venv already needs for every other wrapper. Implemented as a plain PyTorch training loop instead, in-process, no subprocess: calls `load_finetune_splits`, builds train/val/test dataloaders directly from `huggingmolecules`' own `MatFeaturizer`/`RMatFeaturizer`/`GroverFeaturizer` + `MatModel`/`RMatModel`/`GroverModel` (unstripped, unlike `wrapper.py`'s embedding path — their pretrained `generator`/readout heads already default to a single scalar output, `generator_n_outputs`/`readout_n_outputs: int = 1`, so no config surgery is needed for regression), trains with Adam + MSE loss (GROVER: reimplements `GroverLoss`'s two-head-plus-agreement-penalty formula, since `training_loss_fn.py`'s own class needs `gin`'s runtime state initialized), tracks the best validation-MAE epoch, and runs inference on the test split before calling `save_finetune_result`. lr/batch_size/weight_decay/epochs default to whatever each model family's own `experiments/configs/models/{mat,rmat,grover}.gin` documents (not reproduced: those gin configs' `NoamLRScheduler` warmup+decay schedule — a constant LR is used instead). Only the best variant per family is run by default: `mat_masking_20M`, `grover_large`, `rmat_4M`.

### Modal scripts (one per wrapper, mirrors the existing `modal_embed_<name>.py` pattern)

`scripts/modal_finetune_unimol.py`, `modal_finetune_gem.py`, `modal_finetune_selformer.py`, `modal_finetune_huggingmolecules.py` — each builds on the wrapper's existing Modal image (same `init.sh`), and fans out one GPU container per (dataset, model variant) running `python model_wrappers/<name>/finetune.py --dataset <name> --output /finetune_errors/<dataset>/<model>.joblib`.

All four originally created their `/data/{prepared,finetune_errors,finetune_checkpoints}` symlinks unconditionally; Modal reuses a warm container across multiple sequential inputs, so any input beyond the first on a given container crashed with `FileExistsError` (confirmed live: 43/51 huggingmolecules calls failed this way on first rollout). Fixed in all four by checking `os.path.lexists()` first.

## Rollout order

1. **Pilot on Tier 1** — done for `unimol`: `model_wrappers/unimol/finetune.py` + `scripts/modal_finetune_unimol.py`, validated end-to-end on Modal against `reg_openadmet_pxr_pEC50` (5-fold CV via `unimol_tools.MolTrain`, scored via `MolPredict`, result matching the `data/errors/` schema).
2. **Scale Tier 1** across all 17 endpoints — **done for `unimolv1`, `unimolv2`, GEM, SELFormer, and `huggingmolecules`'s best variant per family** (`mat_masking_20M`, `rmat_4M`, `grover_large`; see "Results so far" below). A small hyperparameter sweep for `unimolv2` (lr, epochs) is in progress given its mixed default-hyperparameter results. The remaining `huggingmolecules` variants (`mat_masking_200k`/`2M`, `grover_base`) not yet run at full scale. Chemformer moved to Tier 3, see above.
3. **Tier 2**, CPU-only but small models, using the harness proven in steps 1-2, adapted for local pyenv execution instead of Modal.
4. **Revisit Tier 3** once Tier 1/2 are done — these need real design decisions (what regression head to bolt onto SAND's or CLAMP's contrastive encoders, whether SimSon's downstream path is solid enough to trust) before they're worth investing harness time in.

## Results so far

Full 17-endpoint runs for three Tier 1 models, each at its own library's/script's default hyperparameters (no tuning — see the hyperparameter-sweep note below). Comparison is fine-tuned MAE vs. that same model's own best frozen-embedding+head result (best of rf/ridge/knn/tabicl, from the existing `data/errors/`).

| Model | Wins vs. frozen | Mean MAE improvement | Median |
|---|---:|---:|---:|
| **GEM** | 16/17 | **+22.4%** | +19.4% |
| **Uni-Mol** (`unimolv1`) | 15/17 | +15.7% | +11.1% |
| **SELFormer** | 11/17 | +13.5% | +17.1% |

GEM is the standout — highest win rate and biggest average improvement, despite being CPU-only and needing the trickiest bugfix of the three (a `src` package name collision between this repo and GEM's vendored code, see `model_wrappers/gem/finetune.py`). SELFormer has the lowest win rate; its losses cluster on the smaller ExpansionRx endpoints (Log_Caco_ER, Log_Caco_Papp_AB, Log_HLM_CLint, Log_MLM_CLint) plus one Polaris potency endpoint, plausibly because its defaults (50 epochs, batch_size=8) are the heaviest of the three and may be overfitting small datasets rather than underperforming structurally. The ExpansionRx/Polaris **LogD** endpoints show the single biggest and most consistent win for every model (~40-44%) — a repeated signal, not noise, that frozen embeddings are specifically weak at LogD.

**`unimolv2` — mixed vs. its own `unimolv1` result, at the same default hyperparameters (10 epochs, lr=1e-4, batch_size=16).** 9/17 wins, mean MAE improvement ~0.0%, median +1.2% — a much weaker showing than `unimolv1`'s 15/17 / +15.7% above, despite sharing a wrapper and CLI. The worst outlier, `polaris_admet_LogD` (fine-tuned MAE 0.629 vs. frozen best 0.404, −55.7%), shows an actual training problem rather than plain underperformance: its log shows val_loss oscillating between 0.38-0.50 across epochs with no clean convergence, and train_loss itself isn't monotonically decreasing (0.93→0.66→0.67→0.85) — plausibly `unimolv2`'s much heavier pair-representation architecture needs different hyperparameters (lower lr and/or a smaller batch size for stability) than what already works for v1. A hyperparameter sweep mirroring the Uni-Mol v1 sweep below is in progress. Notably, `expansionrx_LogD` (the other LogD endpoint) still shows a strong +39.5% win, matching the LogD-is-a-repeated-signal pattern noted below — so the regression isn't LogD-specific, it's specific to `polaris_admet_LogD`'s combination of small size (478 rows) and this hyperparameter recipe.

**Hyperparameter sweep (`unimolv2`)**: tripling epochs (10→30) and halving lr (1e-4→5e-5) were each tried on 3 representative endpoints (small/medium/large by row count):

| Endpoint (size) | default MAE | epochs30 MAE | lr5e-05 MAE |
|---|---:|---:|---:|
| `polaris_admet_LogD` (478 rows) | 0.629 | 0.755 (worse) | 0.689 (worse) |
| `pxr_pEC50` (4652 rows) | 0.459 | 0.459 (same) | 0.469 (worse) |
| `expansionrx_LogD` (7309 rows) | 0.362 | 0.336 (+7%) | 0.372 (worse) |

Unlike the Uni-Mol v1 sweep below (where deviating from defaults was merely unhelpful), for v2 it's actively counterproductive on the known bad outlier (`polaris_admet_LogD` got worse under both variants) — this looks like genuine training instability/poor generalization on that dataset, not underfitting fixable by training longer or more gently. The large endpoint saw a small real gain from more epochs, but nothing that changes the overall picture. `batch_size` (left untouched here — fixed the earlier OOM by moving to an A100 instead, see the Tier 1 table above) is the more likely remaining lever, but that's untested; not pursued further for now.

**`huggingmolecules` (MAT/R-MAT/GROVER) — the first Tier 1 family where fine-tuning mostly *doesn't* beat the frozen baseline.** Unlike GEM/Uni-Mol/SELFormer above, all three variants win against their own frozen-embedding+shallow-head result on fewer than half the endpoints:

| Model | Wins vs. frozen | Mean MAE improvement | Median |
|---|---:|---:|---:|
| **R-MAT** (`rmat_4M`) | 8/17 | −10.1% | −6.7% |
| **GROVER** (`grover_large`) | 6/17 | −9.0% | −9.5% |
| **MAT** (`mat_masking_20M`) | 2/17 | −26.6% | −18.3% |

Among the three fine-tuned variants themselves (not vs. frozen), R-MAT is consistently the strongest and GROVER the weakest — e.g. on `reg_openadmet_expansionrx_LogD`, fine-tuned Spearman is 0.87 (R-MAT) / 0.79 (MAT) / 0.81 (GROVER); on several endpoints GROVER's fine-tuned Spearman is near zero or negative (`polaris_admet_HLM`: 0.014, `polaris_admet_LogD`: 0.095, `Log_Mouse_MPB`: −0.024, `potency_pIC50_MERS_CoV_Mpro`: −0.144) despite validation MAE decreasing steadily over training (confirmed from logs — not a training-loop bug, just poor test-set generalization under this repo's default hyperparameters: lr=1e-4, no LR scheduler, batch_size=32). GROVER is also far more expensive per run than MAT/R-MAT (~100s/epoch vs. ~12s/epoch, 20-35 epochs before early-stopping vs. under 10). **Further GROVER-specific tuning (lower lr, more patience, or reproducing the dropped `NoamLRScheduler`) is deferred** — the current defaults-only run stands as the baseline to compare against later.

The `reg_openadmet_polaris_admet_KSOL` row below is a notable outlier (frozen MAE 0.12-0.15 vs. ~0.2-0.25 for unimol/GEM/SELFormer's own frozen baselines on the same endpoint) — confirmed not a scale/leakage bug (same 112-row test set, same molecule_ids, same target scale as the fine-tuned run); more likely this endpoint's small test set (n=112) makes the frozen GridSearchCV-tuned heads unusually lucky for these three specific embedders.

**Hyperparameter sweep (Uni-Mol only)**: tripling epochs (10→30, same lr) moved MAE by under 1% in either direction across 3 representative endpoints (small/medium/large); halving the learning rate made things consistently 2-5% worse. `unimol_tools.MolTrain`'s own defaults already sit at a reasonable operating point — the improvement above is coming from fine-tuning itself, not from hyperparameter tuning, which is good news for not needing an expensive per-endpoint HPO sweep before scaling further.

Per-endpoint numbers (fine-tuned MAE / best-frozen MAE / % improvement):

| Endpoint | Uni-Mol | GEM | SELFormer |
|---|---:|---:|---:|
| ExpansionRx Log_Caco_ER | 0.414 / 0.413 (−0.2%) | 0.368 / 0.434 (+15.2%) | 0.433 / 0.432 (−0.1%) |
| ExpansionRx Log_Caco_Papp_AB | 0.306 / 0.316 (+3.1%) | 0.334 / 0.370 (+9.8%) | 0.369 / 0.358 (−3.0%) |
| ExpansionRx LogD | 0.386 / 0.688 (+43.8%) | 0.432 / 0.721 (+40.1%) | 0.454 / 0.762 (+40.4%) |
| ExpansionRx Log_HLM_CLint | 0.324 / 0.351 (+7.8%) | 0.332 / 0.363 (+8.6%) | 0.394 / 0.359 (−9.6%) |
| ExpansionRx Log_MLM_CLint | 0.370 / 0.397 (+6.9%) | 0.422 / 0.418 (−1.1%) | 0.442 / 0.401 (−10.2%) |
| ExpansionRx Log_Mouse_BPB | 0.173 / 0.227 (+23.6%) | 0.184 / 0.278 (+33.9%) | 0.184 / 0.303 (+39.4%) |
| ExpansionRx Log_Mouse_MPB | 0.225 / 0.257 (+12.4%) | 0.199 / 0.307 (+35.0%) | 0.225 / 0.271 (+17.1%) |
| ExpansionRx Log_Mouse_PPB | 0.210 / 0.267 (+21.5%) | 0.274 / 0.338 (+18.8%) | 0.221 / 0.334 (+33.8%) |
| ExpansionRx KSOL | 0.349 / 0.466 (+25.2%) | 0.368 / 0.558 (+34.0%) | 0.473 / 0.582 (+18.6%) |
| Polaris ADMET HLM | 0.343 / 0.465 (+26.2%) | 0.467 / 0.473 (+1.2%) | 0.453 / 0.480 (+5.5%) |
| Polaris ADMET KSOL | 0.155 / 0.197 (+21.4%) | 0.136 / 0.247 (+45.0%) | 0.212 / 0.211 (−0.7%) |
| Polaris ADMET LogD | 0.497 / 0.549 (+9.6%) | 0.485 / 0.602 (+19.4%) | 0.671 / 0.720 (+6.8%) |
| Polaris ADMET MDR1-MDCKII | 0.204 / 0.300 (+32.1%) | 0.203 / 0.368 (+45.0%) | 0.281 / 0.392 (+28.2%) |
| Polaris ADMET MLM | 0.369 / 0.409 (+9.7%) | 0.423 / 0.478 (+11.6%) | 0.445 / 0.619 (+28.1%) |
| Polaris Potency pIC50 MERS | 0.640 / 0.638 (−0.4%) | 0.586 / 0.656 (+10.6%) | 0.624 / 0.596 (−4.6%) |
| Polaris Potency pIC50 SARS | 0.712 / 0.751 (+5.2%) | 0.589 / 0.766 (+23.1%) | 0.574 / 0.700 (+18.0%) |
| PXR pEC50 | 0.467 / 0.650 (+28.1%) | 0.525 / 0.758 (+30.8%) | 0.588 / 0.745 (+21.1%) |

Per-endpoint numbers for `unimolv2` (fine-tuned MAE / that model's own best-frozen MAE / % improvement):

| Endpoint | Uni-Mol v2 |
|---|---:|
| ExpansionRx Log_Caco_ER | 0.414 / 0.387 (−6.9%) |
| ExpansionRx Log_Caco_Papp_AB | 0.347 / 0.352 (+1.2%) |
| ExpansionRx LogD | 0.362 / 0.598 (+39.5%) |
| ExpansionRx Log_HLM_CLint | 0.284 / 0.326 (+12.8%) |
| ExpansionRx Log_MLM_CLint | 0.374 / 0.401 (+6.5%) |
| ExpansionRx Log_Mouse_BPB | 0.197 / 0.231 (+14.6%) |
| ExpansionRx Log_Mouse_MPB | 0.267 / 0.225 (−18.6%) |
| ExpansionRx Log_Mouse_PPB | 0.234 / 0.272 (+14.0%) |
| ExpansionRx KSOL | 0.381 / 0.474 (+19.6%) |
| Polaris ADMET HLM | 0.402 / 0.369 (−8.9%) |
| Polaris ADMET KSOL | 0.197 / 0.172 (−14.5%) |
| Polaris ADMET LogD | 0.629 / 0.404 (−55.7%) |
| Polaris ADMET MDR1-MDCKII | 0.299 / 0.259 (−15.6%) |
| Polaris ADMET MLM | 0.419 / 0.396 (−5.9%) |
| Polaris Potency pIC50 MERS | 0.642 / 0.618 (−3.8%) |
| Polaris Potency pIC50 SARS | 0.683 / 0.727 (+6.1%) |
| PXR pEC50 | 0.459 / 0.539 (+15.0%) |

Per-endpoint numbers for `huggingmolecules` (fine-tuned MAE / that variant's own best-frozen MAE / % improvement):

| Endpoint | MAT (`mat_masking_20M`) | R-MAT (`rmat_4M`) | GROVER (`grover_large`) |
|---|---:|---:|---:|
| ExpansionRx Log_Caco_ER | 0.463 / 0.403 (−15.0%) | 0.391 / 0.412 (+5.3%) | 0.544 / 0.467 (−16.6%) |
| ExpansionRx Log_Caco_Papp_AB | 0.354 / 0.283 (−25.4%) | 0.342 / 0.302 (−13.2%) | 0.486 / 0.415 (−17.1%) |
| ExpansionRx LogD | 0.464 / 0.520 (+10.9%) | 0.407 / 0.498 (+18.4%) | 0.482 / 0.786 (+38.7%) |
| ExpansionRx Log_HLM_CLint | 0.344 / 0.314 (−9.4%) | 0.311 / 0.314 (+1.0%) | 0.383 / 0.379 (−1.0%) |
| ExpansionRx Log_MLM_CLint | 0.443 / 0.375 (−18.3%) | 0.381 / 0.357 (−6.7%) | 0.376 / 0.415 (+9.5%) |
| ExpansionRx Log_Mouse_BPB | 0.297 / 0.194 (−52.8%) | 0.193 / 0.204 (+5.6%) | 0.354 / 0.322 (−9.8%) |
| ExpansionRx Log_Mouse_MPB | 0.268 / 0.201 (−33.4%) | 0.227 / 0.194 (−17.3%) | 0.386 / 0.331 (−16.5%) |
| ExpansionRx Log_Mouse_PPB | 0.324 / 0.277 (−17.0%) | 0.192 / 0.273 (+29.5%) | 0.336 / 0.312 (−7.7%) |
| ExpansionRx KSOL | 0.410 / 0.469 (+12.5%) | 0.409 / 0.427 (+4.2%) | 0.507 / 0.560 (+9.4%) |
| Polaris ADMET HLM | 0.425 / 0.357 (−19.0%) | 0.402 / 0.347 (−15.9%) | 0.493 / 0.495 (+0.5%) |
| Polaris ADMET KSOL | 0.292 / 0.128 (−127.1%) | 0.268 / 0.119 (−125.7%) | 0.275 / 0.154 (−78.3%) |
| Polaris ADMET LogD | 0.524 / 0.485 (−8.1%) | 0.471 / 0.515 (+8.5%) | 0.849 / 0.776 (−9.5%) |
| Polaris ADMET MDR1-MDCKII | 0.228 / 0.181 (−25.7%) | 0.272 / 0.218 (−24.6%) | 0.585 / 0.445 (−31.3%) |
| Polaris ADMET MLM | 0.388 / 0.360 (−7.8%) | 0.478 / 0.406 (−17.9%) | 0.577 / 0.580 (+0.6%) |
| Polaris Potency pIC50 MERS | 0.845 / 0.528 (−59.9%) | 0.622 / 0.562 (−10.7%) | 0.935 / 0.843 (−10.9%) |
| Polaris Potency pIC50 SARS | 0.946 / 0.612 (−54.5%) | 0.751 / 0.636 (−18.0%) | 1.475 / 1.207 (−22.1%) |
| PXR pEC50 | 0.542 / 0.534 (−1.4%) | 0.490 / 0.524 (+6.3%) | 0.655 / 0.726 (+9.8%) |

## Future idea: fine-tuning per challenge (multi-task)

Rather than one fine-tuned model per endpoint (the current approach) or one model spanning all 17 endpoints, a middle option worth revisiting once single-task Tier 1 results are in: one shared-encoder model **per data source/challenge**, with a multi-dim output head and a masked loss (a compound only has labels for its own source's endpoints):

- ExpansionRx's 9 endpoints jointly (they share one compound set).
- Polaris's 7 endpoints jointly — the 5 ADMET + 2 Potency endpoints share one antiviral compound series.
- PXR's 1 endpoint alone (no siblings to share with).

This is cheaper to train (3 models instead of 17) and lets same-source endpoints transfer signal to each other (they're correlated, measured on the same molecules). The expected downside: cross-source transfer should be weak, since ExpansionRx/Polaris/PXR are essentially disjoint chemical series from different programs, so a single joint encoder mostly helps within a source, not across all three. It also changes `docs/bayesian_model.md`'s per-endpoint independence assumption somewhat — endpoints sharing one fine-tuned encoder are more correlated than endpoints that only share a frozen embedding — though this isn't a bigger departure than today's frozen-embedding setup, which already shares one encoder across all 17 per-endpoint sklearn heads. Not implemented yet; the single-task per-endpoint runs above remain the baseline to compare against.

## Comparison / analysis

Once both a frozen-embedding result (`data/errors/`) and a fine-tuned result (`data/finetune_errors/`) exist for the same (dataset, model), add a small comparison step — either a new script or an extension of `scripts/blam_analysis.py`/`scripts/visualize_blam.py` — producing a per-endpoint delta table: fine-tuned MAE/Spearman vs. that model's best frozen-embedding+head result. This directly answers, per model, whether end-to-end fine-tuning beats treating it as a frozen embedding under this benchmark's data and splits.

## Open items / non-goals

- Chemeleon and chemprop-family models are out of scope here — the underlying models were built in `../ensemble-aware-admet`; bringing them to this repo's 17 endpoints is planned in `docs/e2e_chemprop_chemeleon_plan.md`.
- Tier 3 wrappers (`SAND`, `clamp`, `simson`, `mol-r`) need design decisions before implementation — not just script invocation — and should be revisited after the Tier 1/2 harness is proven out.
- `coati` is deprioritized pending a decision on whether building a from-scratch training loop (no existing per-task fine-tune entry point) is worth the investment relative to the other candidates.
- The Modal cost/duration of Tier 1 training runs (heavier than the existing embedding-only Modal jobs) hasn't been measured yet — the pilot step above should establish this before scaling to all 17 endpoints.
- GROVER's fine-tuning underperformance (see "Results so far") is left as-is for now — no hyperparameter tuning attempted yet. Worth revisiting: lower lr, more patience, or reproducing the dropped `NoamLRScheduler` warmup+decay schedule from `experiments/configs/models/grover.gin`.
- `unimolv2`'s mixed default-hyperparameter results (see "Results so far") were followed up with a small lr/epochs sweep on 3 representative endpoints — neither knob helped (one made the known bad outlier worse). `batch_size` is the more likely remaining lever, untested so far — worth a follow-up if `unimolv2` is going to be invested in further, same open question as GROVER above.
- The remaining `huggingmolecules` variants (`mat_masking_200k`/`2M`, `grover_base`) haven't been run at full scale yet.
