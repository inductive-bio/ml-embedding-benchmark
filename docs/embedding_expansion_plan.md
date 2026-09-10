# Embedding expansion plan

## Context

The benchmark currently only covers the ~15 models used in the original paper (fingerprints, Mol2Vec, CDDD, Chemformer, COATI, CLAMP, MolBERT, SimSon, various HF LMs, MAT/R-MAT/Grover, GEM, Uni-Mol v1/v2, GraphMVP+baselines, GraphFP, mol-r). Reviewing them turned up a real gap: only Uni-Mol is a genuine conformer-based 3D GNN at inference time; GEM is 2D+3D distilled, and everything else is 2D-only. To make the benchmark comprehensive, and to give the downstream Bayesian ranking model (`docs/bayesian_model.md`) a wider and more informative set of per-endpoint error statistics to compare, we're adding:

1. Simple, cheap descriptor-based baselines (RDKit 2D, Mordred 2D, RDKit 912-dim 3D, Mordred 2D+3D, and a Mordred+RDKit-3D concatenation) that establish a floor for how much value the more complex learned embeddings add.
2. A placeholder section (below) for genuinely 3D-at-inference pretrained GNN candidates — under separate investigation.

Investigation of `../ensemble-aware-admet` (a sibling repo with related descriptor/embedding code) found that its `chemeleon_3d.py`, `denoise3d_3d.py`, and `loqi_3d.py` are **not** true 3D models: all three are plain 2D chemprop MPNNs distilled to predict 3D-derived targets, so porting them would not close the true-3D gap (same category as GEM). LoQI additionally has no working local execution path (Modal-only, incompatible torch/torch-geometric pin) and its own docs note its embedding is "almost entirely graph-driven." These are excluded from this pass.

Chemprop (random-init or pretrained, trained end-to-end per endpoint) is also deferred — it doesn't fit the existing frozen-embedding + sklearn-head architecture and needs its own evaluation path. Worth a separate follow-up plan once the descriptor baselines land; the per-endpoint error statistics it would produce are a sufficient interface into the existing Bayesian ranking model even without going through the `Embedder` ABC.

## Scope for this pass

Implemented as **8 model variants** in a single `model_wrappers/descriptors/`, not the 3 separate wrappers originally sketched below (see "Implementation" for why). Two of the eight are conformer-free; the other six ablate the three conformer-dependent descriptor families across two levels of theory:

1. **`rdkit2d`** — 217 raw `rdkit.Chem.Descriptors.descList` values (no conformer). This is the same descriptor set `deepchem.feat.RDKitDescriptors()` wraps by default (confirmed identical count); implemented directly against rdkit to avoid adding deepchem as a dependency for this alone.
2. **`mordred2d`** — Mordred 2D-only descriptors (`ignore_3D=True`), ~1613-dim, no conformer.
3. **`rdkit3d_etkdg`, `rdkit3d_mmff94s`** — the 912-dim RDKit 3D featurizer (shape descriptors + PBF + WHIM + GETAWAY + AUTOCORR3D + RDF + MORSE), one variant per level of theory.
4. **`mordred3d_etkdg`, `mordred3d_mmff94s`** — Mordred 2D+3D (`ignore_3D=False`), ~1826-dim, one variant per level of theory.
5. **`mordred2d_rdkit3d_etkdg`, `mordred2d_rdkit3d_mmff94s`** — Mordred 2D-only concatenated with the 912-dim RDKit 3D set (not Mordred's own 3D descriptors, to avoid duplicating 3D information), ~2525-dim, one variant per level of theory.

**Levels of theory** for the conformer-dependent variants (`src/common/conformers.py`):
- `etkdg` — a single raw ETKDG distance-geometry embedding, no force-field refinement.
- `mmff94s` — a 5-conformer ETKDG ensemble, each optimized with MMFF94s (UFF fallback), lowest-energy conformer kept.

A third level (ETKDG + GFN2-xTB semiempirical refinement) was considered — `xtb` is already installed on this machine and the sibling repo has a working `gfn2_optimize()` to port — but was left out of this pass; add it as a third suffix (`_gfn2xtb`) per conformer-dependent family if wanted later.

## True-3D GNN candidates (placeholder — under separate research)

None of the pretrained 3D-ish models found in the sibling repo actually consume coordinates at inference. Uni-Mol remains the only genuine 3D GNN in the benchmark today. Candidates for a real second true-3D embedder are being researched separately and will be added here, e.g.:

- Model name / checkpoint source:
- Genuinely 3D at inference (yes/confirmed how):
- Dependencies beyond what's already in `requirements.txt`:
- Integration notes (fits `SmilesEmbedder` directly vs. needs a new `Embedder` subtype):

## Architecture fit

All 8 variants are plain fixed-length feature vectors, so they fit the existing `SmilesEmbedder` pattern with zero changes to core pipeline code (`src/common/types.py`, `src/embedding/embedding.py`). Deviating from the original per-embedder-directory sketch above, all 8 share **one** `model_wrappers/descriptors/` (`wrapper.py`, `.python-version`, `init.sh`) rather than 3+ separate wrapper dirs — the 3 conformer-dependent families all depend on rdkit + mordredcommunity, and there was no reason to pay 3 separate venv installs for that. (Cross-variant conformer *reuse* is a separate story — see the Modal section below; the local disk cache described here helps local/Hydra runs but each Modal container is a fresh process, so the Modal path shares conformers a different way, within a batch rather than across containers.)

- `wrapper.py` implements `forward(self, smiles) -> np.ndarray` of shape `(N, D)`, filling failed rows with `np.nan` (never dropping — this is what `EmbeddedDataset.remove_failed_embeddings()` expects, matching the `safe_batch`/`safe_step` pattern already used in `model_wrappers/huggingmolecules/wrapper.py`, `model_wrappers/unimol/wrapper.py`, `model_wrappers/gem/wrapper.py`).
- Exposes a module-level `get_embedder(name, **kwargs)` factory, dispatching on an exact-match dict over the 8 model names (not substring matching — `mordred2d_rdkit3d_etkdg` contains both `mordred2d` and `rdkit3d_etkdg` as substrings, which would be ambiguous).
- `config/experiment/descriptors.yaml` sweeps `+model_name` over all 8 (`huggingmolecules.yaml`'s pattern, not `fingerprints.yaml`'s per-variant `config/model/*.yaml` files — no reason to create 8 one-line YAML files when a single inline list works).

Both `unimol` and `gem` wrappers already generate conformers internally inside `forward()` (via `unimol_tools`'s own `AllChem.EmbedMolecule`, and `gem_repo`'s `DownstreamTransformFn` respectively) — so hiding conformer generation inside `forward()` is the established pattern, not a new one.

## Implementation

### Conformer generation (shared utility, `src/common/conformers.py`)

Ported from `ensemble-aware-admet/conformer_generation/ensemble_generation.py`. Exposes the two levels of theory directly, each disk-cached (`joblib.Memory`, keyed by SMILES + params) so the local/Hydra path doesn't regenerate the same molecule's conformer across the 3 dependent variants x 2 levels:

- `first_etkdg_conformer(smiles)` — single `AllChem.EmbedMolecule(randomSeed=42)`, no refinement. Deliberately not `create_etkdg_ensemble(...)[0]`: that generates 5 conformers to give MMFF something to rank, which is wasted work when nothing here ranks them.
- `best_mmff94s_conformer(smiles, num_confs=5, rms_threshold=0.25)` — `create_etkdg_ensemble` (`AllChem.EmbedMultipleConfs`) + `mmff_optimize` (MMFF94s via `MMFFGetMoleculeForceField`, UFF fallback, drop conformers where `ff.Minimize()` doesn't converge), lowest-energy pick.
- Both raise on total failure (no conformer embeds, e.g. disconnected fragments, or nothing converges) — the caller NaN-fills that row rather than crashing the batch.

### `model_wrappers/descriptors/rdkit_3d.py`

`RDKit3DFeaturizer`, ported verbatim from `ensemble-aware-admet/src/models/rdkit_3d.py` (Descriptors3D shape descriptors + `CalcPBF` + `CalcWHIM` + `CalcGETAWAY` + `CalcAUTOCORR3D` + `CalcRDF` + `CalcMORSE` = 912 features), operating on a `Chem.Mol` that already carries one conformer.

### `model_wrappers/descriptors/wrapper.py`

One `DescriptorEmbedder` base class with a shared `forward()`/`_row_or_nan()` loop; 5 subclasses (`RDKit2DEmbedder`, `Mordred2DEmbedder`, `RDKit3DEmbedder`, `Mordred3DEmbedder`, `Mordred2DRDKit3DEmbedder`, the latter two taking a `conformer_level` argument) cover all 8 model names via the `_EMBEDDERS` dict.

**A real bug worth flagging for anyone touching this again**: composite embedders (`Mordred2DRDKit3DEmbedder`) must let a sub-embedder's exception *propagate* to the outer per-molecule try/except, not catch it and return a partial-NaN array. Catching it internally makes a hard failure (unparsable SMILES, conformer embedding failure) look like a *successful* computation with legitimately-missing descriptors — which then gets zero-filled (see next point) into a fake all-zero row instead of being dropped.

**Another real bug, this one about Mordred specifically**: Mordred marks per-descriptor values it can't compute for a given molecule (e.g. `MAXsLi` when there's no Lithium atom) as `Missing`, not an error — true of *some* descriptor for nearly every real molecule (verified against real OpenADMET compounds: 150-370 missing descriptors out of 1613, even for small drug-like molecules like ibuprofen or aspirin). `EmbeddedDataset.remove_failed_embeddings()` drops any row with *any* NaN, not just all-NaN rows, so leaving these as NaN would silently drop ~100% of rows for every Mordred-based variant. Fixed centrally in `_row_or_nan()`: a successful computation's individual NaNs get zeroed (0.0 is also the natural value for "this element/environment contributes nothing"); only a hard failure (exception) NaN-fills the *whole* row, which is what should get dropped.

### Configs

`config/experiment/descriptors.yaml`, sweeping `+model_name` over the 8 variant names.

## Open item (tracked, not in this pass)

`docs/openadmet_migration_status.md` already flagged that `src/eval/supervised/train.py`/`eval_metrics.py` hardcode classification metrics regardless of `task`. That's since been addressed by a separate regression-specific path (`src/eval/supervised/regression.py::fit_regressor` + `scripts/compute_prediction_errors.py`), used to score these embeddings — see "Scoring" below.

## Verification

Confirmed via `embed.py` end-to-end against a real OpenADMET dataset (`reg_openadmet_polaris_admet_MLM`, 420 rows): correct dimensionality per variant, exactly the genuinely-unembeddable rows dropped (not the ~100% that the Mordred NaN bug above would have caused), and per-molecule timing that motivated moving the full run to Modal (see below): rdkit2d and rdkit3d essentially instant per molecule; `mordred3d_etkdg` ~0.45s/molecule (Mordred's full 2D+3D descriptor DAG is the cost, not conformer generation or Calculator construction — both independently confirmed fast).

## Running at scale: Modal

45k molecule-rows x 8 variants, with 3 of the 8 needing conformer generation, doesn't fit in a reasonable wall-clock time on this machine's 8 cores. `scripts/modal_embed_descriptors.py` fans this out across many Modal containers at (dataset, model_name, batch-of-200-molecules) granularity — finer than `scripts/modal_embed.py`'s one-container-per-wrapper grain or even `scripts/modal_score_errors.py`'s one-container-per-(embedding,method,endpoint) grain, because a whole-dataset unit here would let the biggest ExpansionRx datasets (~7300 rows) dominate wall-clock on the slowest variants even with every other combo already done (same rationale `modal_score_errors.py`'s own docstring gives for its grain choice). Each batch container returns its embedding matrix directly; the local entrypoint concatenates batches in order and makes one final per-(dataset, model_name) call to assemble the `EmbeddedDataset` (needs the prepared `Dataset`'s splits/labels) and write it to the Volume.

    modal run --detach scripts/modal_embed_descriptors.py
    modal volume get ml-embedding-benchmark /embedded data/embedded

## Scoring

Once embedded, score with the existing regression path (not Hydra-based, see `docs/openadmet_migration_status.md`):

    python scripts/compute_prediction_errors.py --embedding rdkit2d --method rf
    modal run --detach scripts/modal_score_errors.py --embeddings rdkit2d,mordred2d,rdkit3d_etkdg,rdkit3d_mmff94s,mordred3d_etkdg,mordred3d_mmff94s,mordred2d_rdkit3d_etkdg,mordred2d_rdkit3d_mmff94s

### Bug found and fixed while scoring: row-alignment after dropped embeddings

`compute_errors()` (`scripts/compute_prediction_errors.py`) required `embedded.X`'s row count to exactly equal the full prepared dataset's row count, skipping (no error, no output, exit 0) whenever they differed — its own comment claimed `remove_failed_embeddings()` "doesn't persist which original rows survived." That's incorrect: `remove_failed_embeddings()` filters `self.y` via `.iloc[valid_indices]`, and pandas' `.iloc` preserves each surviving row's *original* index label rather than resetting it — confirmed empirically (`embedded.y.index` on a dataset with dropped rows contained exactly the original 0..N-1 positions that survived). Our conformer-dependent variants have a nonzero drop rate on most datasets (ETKDG genuinely can't embed some bridged/fused bicyclic scaffolds), which meant most of their (dataset, method) scoring combos were being silently skipped rather than computed — 126 of 408 for our 8 embeddings alone. Fixed by using `embedded.y.index` to select the matching `molecule_id`/target from the full prepared dataset instead of requiring an exact row-count match; this is also simpler, since it's correct in the common (nothing dropped) case too, not just a special-cased mismatch branch.

This same gap likely silently skipped scoring combos for other pre-existing embeddings too, anywhere they had a nonzero embed-time drop rate on some dataset — not something this pass re-ran for the full 57-embedding set, only for the 8 new ones.
