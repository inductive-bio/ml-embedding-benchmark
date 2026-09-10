# End-to-end chemprop + chemeleon-style training plan

## Context

`docs/finetuning_plan.md` covers fine-tuning wrappers that ship their own fine-tune code (Uni-Mol, GEM, SELFormer, huggingmolecules, molbert, graphmvp) on the 17 OpenADMET regression endpoints (`config/dataset/reg_openadmet_*.yaml`), writing results to `data/finetune_errors/` alongside the frozen-embedding results in `data/errors/`. That doc explicitly excluded `chemeleon` and any chemprop-family model, deferring them to the sibling repo `../ensemble-aware-admet`, where two "standard" model families already exist, targeting ExpansionRx's 9 endpoints only:

- **`chemprop-2d`** (`baselines/chemprop_baseline.py`) — a chemprop `MPNN` (BondMessagePassing + aggregation + RegressionFFN) trained from scratch directly on ADMET labels, with RDKit-2D descriptors as extra per-molecule features.
- **`chemprop-3d`** (`src/models/chemprop_3d.py`) — same architecture, RDKit-3D descriptors (912-dim, from a conformer) instead of/alongside 2D.
- **Chemeleon-style descriptor pretraining + fine-tuning** (`src/models/chemeleon_3d.py`, `chemeleon_finetune.py`) — a chemprop MPNN pretrained via masked descriptor-prediction (RDKit-3D and/or Mordred-2D targets) on large unlabeled corpora (PubChem/GEOM-Drugs/ChEMBL3D), then fine-tuned by swapping in a fresh regression head on ADMET labels.

This plan brings the same two model families to bear on **all 17 OpenADMET endpoints already tracked in this repo** (ExpansionRx ×9, Polaris antiviral ADMET+Potency ×7, PXR ×1), reusing this repo's existing fine-tuning harness so results land in `data/finetune_errors/` next to the other fine-tuned models for direct comparison. A follow-up step — running an MCMC-based comparison across all now-available methods — is explicitly out of scope here; this doc only covers getting the standard chemprop/chemeleon results computed and saved.

This doc is a plan, not an implementation — no code exists yet for the items below.

## What's already in place to reuse

- **The fine-tuning harness**: `src/common/finetune_data.py::load_finetune_splits(dataset_name, target_column) -> (train_df, val_df, test_df)` and `src/common/finetune_results.py::save_finetune_result(dataset_name, model_name, molecule_id, y_true, y_pred, metric_name, ...)`, writing the same `{molecule_id, y_true, y_pred, error, mae, spearmancorr}` schema `scripts/compute_prediction_errors.py` writes to `data/errors/` — `scripts/blam_analysis.py` needs no changes to pick up new results.
- **3D conformers for all 17 endpoints, already generated**: `data/conformers/{expansionrx,polaris_antiviral,pxr}_mmff94s.joblib` — up to 50 ETKDG conformers/molecule (RMSD-pruned at 0.1 Å) + MMFF94s energies, produced by `scripts/modal_generate_conformers.py` + `modal_optimize_conformers.py` via `src/common/conformers.py::create_etkdg_ensemble`/`mmff_optimize` (ported from ensemble-aware-admet). Keyed by canonical SMILES, deduplicated per **challenge**. Dataset → challenge mapping already defined (`CHALLENGES`/`CHALLENGE_DATASET_GLOBS` in those scripts): `reg_openadmet_expansionrx_*` → `expansionrx`, `reg_openadmet_polaris_admet_*`/`polaris_potency_*` → `polaris_antiviral`, `reg_openadmet_pxr_*` → `pxr`. No new conformer generation needed.
- **RDKit-3D featurization (912-dim), already ported**: `model_wrappers/descriptors/rdkit_3d.py::compute_rdkit_3d(mol)`.
- **Chemeleon pretrained checkpoints, already downloaded**: `model_wrappers/chemeleon/checkpoints/` has all 7 variants (`chemeleon_original` + 6 ensemble-aware-admet-pretrained variants across PubChem/ChEMBL3D/GEOM-Drugs × {3D-only, 3D+Mordred2D} × {mean, Boltzmann}), pulled via `init.sh` from the `ensemble-aware-admet` Modal volume. `model_wrappers/chemeleon/wrapper.py::_load_mpnn` already knows how to load either checkpoint format into a full `MPNN`.

## Scope

Nine model families, each trained/fine-tuned independently per endpoint (17 endpoints × 9 families = 153 runs total):

1. `chemprop_2d` — MPNN trained from scratch, RDKit-2D descriptors as extra features.
2. `chemprop_3d` — MPNN trained from scratch, RDKit-3D descriptors (from the lowest-energy cached MMFF94s conformer) as extra features.
3. `chemeleon_original`, `chemeleon_pubchem_3d_mean`, `chemeleon_pubchem_3d_boltzmann`, `chemeleon_pubchem_both_mean`, `chemeleon_pubchem_both_boltzmann`, `chemeleon_chembl3d_both_boltzmann`, `chemeleon_geomdrugs_both_boltzmann` — each already-pretrained checkpoint's encoder, fine-tuned end-to-end (encoder + fresh regression head, no freezing) on ADMET labels.

**Single-task per endpoint**: 17 independent trainings per model family, matching every other wrapper's per-endpoint comparison table in `docs/finetuning_plan.md`, rather than ensemble-aware-admet's native multi-task-per-source setup (that would need new masked-loss/multi-task data-loading code this repo doesn't have, and would change `docs/bayesian_model.md`'s per-endpoint independence assumption — see "Future idea: fine-tuning per challenge (multi-task)" in `docs/finetuning_plan.md`, already flagged there as unimplemented).

Not in scope: new chemeleon pretraining, a combined chemprop-2d+3d variant, multi-task/joint-loss training, GFN2-xTB-refined conformers (sticks with the already-cached ETKDG+MMFF94s ensembles, per `docs/embedding_expansion_plan.md`'s decision not to port that level of theory), and any MCMC-based or other statistical model-comparison step.

## New files

### `model_wrappers/chemprop/` (new wrapper directory — chemprop was never wrapped as a frozen embedder here, only used inside `chemeleon`)

- **`init.sh`** — mirrors `model_wrappers/chemeleon/init.sh`'s pins (`torch==2.5.1`, `chemprop==2.2.4`) minus the checkpoint download (chemprop-2d/3d train from scratch, no pretrained weights).
- **`finetune.py`** — CLI: `--dataset <config_name> --features {2d,3d} --output-dir ... --epochs --batch-size --lr --seed --run-tag --override`, matching the `--dataset --output [...]` shape every existing `finetune.py` uses. Steps:
  1. `load_finetune_splits(dataset_name, target_column)` for train/val/test.
  2. Build `x_d` per molecule: `--features 2d` → chemprop's own `RDKit2DFeaturizer()` (217-dim, no conformer needed). `--features 3d` → look up the dataset's challenge (same mapping as `scripts/modal_generate_conformers.py::CHALLENGES`), load `data/conformers/<challenge>_mmff94s.joblib`, pick the lowest-energy conformer per molecule, reconstruct a 3D `Chem.Mol` (same coordinate-setting pattern `scripts/modal_optimize_conformers.py` already uses), featurize with `compute_rdkit_3d` (912-dim). Molecules missing from the conformer cache are dropped (log the count).
  3. Fit a `sklearn.preprocessing.RobustScaler` on train's `x_d`, clip ±10, apply to val/test (matching ensemble-aware-admet's own recipe).
  4. Build `MoleculeDatapoint.from_smi(smiles, y=[value], x_d=...)` → `MoleculeDataset` → `build_dataloader` per split.
  5. Model: `BondMessagePassing` + `NormAggregation` + `RegressionFFN(n_tasks=1, input_dim=mp.output_dim + x_d_width)`, wrapped in chemprop's `MPNN`. Single task, no masked loss needed (unlike ensemble-aware-admet's 9-task ExpansionRx setup — every row here already has a value).
  6. Train with a plain PyTorch Lightning `Trainer` (`ModelCheckpoint(monitor="val_loss")`, `EarlyStopping(monitor="val_loss")`), starting hyperparameters from ensemble-aware-admet's own `configs/chemprop_3d.yaml` / `baselines/configs/chemprop_2d_small.yaml` as defaults — no full HPO sweep initially, consistent with `docs/finetuning_plan.md`'s stated approach for every other wrapper.
  7. Predict on test, call `save_finetune_result(dataset_name, model_name=f"chemprop_{features}", ...)`.

### `model_wrappers/chemeleon/finetune.py` (new, alongside the existing `wrapper.py`)

- CLI: `--dataset <config_name> --model-name {chemeleon_original,...} --output-dir ... --epochs --lr --run-tag --override`.
- Loads the pretrained encoder for the given checkpoint variant (factor `wrapper.py::_load_mpnn`'s loading logic into a small shared helper returning just `.message_passing`, reused by both `wrapper.py` and `finetune.py`).
- Attaches a **fresh** `RegressionFFN(n_tasks=1, input_dim=encoder.output_dim)` via `MeanAggregation()`, wraps in `MPNN`; fine-tunes the whole model (encoder + head) jointly — no freezing, matching ensemble-aware-admet's `chemeleon_finetune.py`.
- Same `load_finetune_splits` → Lightning `Trainer` (checkpoint + early stopping) → `save_finetune_result` pattern as `chemprop/finetune.py`, graph-only (no extra `x_d`), starting hyperparameters from ensemble-aware-admet's `configs/chemeleon_3d_rx.yaml`.

### `scripts/modal_finetune_chemprop.py`, `scripts/modal_finetune_chemeleon.py`

Mirror the existing `scripts/modal_finetune_unimol.py` pattern: one Modal GPU container per (dataset, features-or-checkpoint-variant) run, `.spawn()`-fanned, with the `os.path.lexists()` symlink-creation guard already fixed into the other four Modal finetune scripts (warm-container reuse causes `FileExistsError` otherwise).

## Rollout order

1. **Pilot** on one small endpoint (e.g. `reg_openadmet_polaris_admet_KSOL`, 112 test rows): `chemprop_2d`, then `chemprop_3d` (validates conformer lookup + missing-molecule handling), then one chemeleon variant (`chemeleon_original`) end-to-end — confirm `data/finetune_errors/` output matches the existing schema before scaling.
2. **Scale `chemprop_2d`/`chemprop_3d`** to all 17 endpoints.
3. **Scale chemeleon fine-tuning** across all 7 checkpoint variants × 17 endpoints (119 runs) — the bulk of the compute; fan out via Modal.
4. Cross-reference this doc from `docs/finetuning_plan.md`'s "Excluded per user instruction" / "Open items" sections (done as part of this plan — see that doc).

## Comparison / analysis (future work, not covered here)

Once `data/finetune_errors/` has results for `chemprop_2d`, `chemprop_3d`, and all 7 chemeleon variants across all 17 endpoints, the next step — run separately — is an MCMC-based comparison across all now-available methods (frozen-embedding+head, Tier 1/2 fine-tuned wrappers, and these chemprop/chemeleon results) to determine which perform best. That comparison methodology is intentionally not designed in this doc.

## Open items / non-goals

- No new chemeleon pretraining — reuses the 7 already-downloaded checkpoints as-is.
- No multi-task/joint-loss training.
- No MCMC-based or other statistical model-comparison step (future work, see above).
- No GFN2-xTB-refined conformers.
- Hyperparameters start from ensemble-aware-admet's own defaults; no per-endpoint HPO sweep planned initially, same rationale as `docs/finetuning_plan.md`.
- 153 total runs (17 × 9) is a significant compute ask — the pilot step should establish per-run cost/duration before scaling to all endpoints and all 7 chemeleon variants.
