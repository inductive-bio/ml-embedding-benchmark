# OpenADMET PXR Induction Challenge — Data Guide

> **Status:** downloaded and cleaned by `scripts/prepare_openadmet_data.py`. Raw pulls
> in `data/raw/pxr/`, cleaned `pEC50` endpoint CSV in `data/processed/pxr/pEC50.csv`.
> Confirmed: train=4139 rows (not 4140 as roughly stated in the tutorial notebook's
> printed cell — off by one, immaterial), phase1=253 + phase2=260 = 513 test rows
> exactly as expected. Both phase-unblinded files already have a proper `SMILES`
> column (matching train) — no `CXSMILES (CDD Compatible)` rename needed; that
> quirk only affects the fully-blinded file this guide doesn't use.

Source tutorial reviewed: `data_repos/PXR-Challenge-Tutorial`
(`notebooks/activity_prediction.ipynb`, `evaluation/`, `validation/`).

This repo has **two tracks**: Activity (predict pEC50 from SMILES — relevant to us) and
Structure (predict/dock 3D poses against a PXR receptor structure — not relevant to an
embedding benchmark, ignore `notebooks/structure_prediction.ipynb`,
`inputs/*.cif/.fasta`, and `outputs/example_structure_submission/`). This guide only
covers the Activity track.

Like the other two challenges, PXR has already concluded (ran March–July 2026, final
deadline 2026-07-01) and the test set has since been fully unblinded across two phase
files — no live blind submission needed.

## 1. Data source

HF dataset repo: **`openadmet/pxr-challenge-train-test`**. Files confirmed to exist in
this repo (via the HF API), relevant ones for the Activity track:

```python
import pandas as pd

train_df = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv")

# The tutorial only loads this one — it has NO labels (live-challenge blinded test):
# test_blinded_df = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_BLINDED.csv")

# Use these two instead — together they cover all 513 blinded test compounds, now with real pEC50:
test_phase1_df = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
test_phase2_df = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
test_df = pd.concat([test_phase1_df, test_phase2_df], ignore_index=True)
```

Phase 1 (253 compounds) was unblinded mid-challenge (2026-05-26); Phase 2 (260
compounds) was unblinded at challenge close (2026-07-01). `253 + 260 = 513`, matching
the tutorial's stated blinded test set size — confirm this once you actually load the
files, but the counts line up. No auth required (public dataset).

Two auxiliary train-side files the tutorial also loads but the notebook explicitly does
**not** use in its baseline model — decide whether they're worth incorporating:
- `pxr-challenge_counter-assay_TRAIN.csv` — PXR-null cell line readouts for the same
  compounds, useful for filtering out non-specific-activity hits (see §4).
- `pxr-challenge_single_concentration_TRAIN.csv` — a primary single-concentration
  screen (~21k rows across 4 concentrations) used for hit triage before the
  dose-response follow-up; noisier and off-target-scale (`log2_fc_estimate`, not
  pEC50), probably not directly usable as regression training data for this benchmark's
  single-task-regression setup.

## 2. Schema (main training file, `pxr-challenge_TRAIN.csv`)

| Column | Notes |
|---|---|
| `Molecule Name` | e.g. `OADMET-0006089` |
| `SMILES` | plain SMILES |
| `OCNT Batch` | assay batch ID, not a molecule property — drop |
| `pEC50` | **primary regression target**, already log-scale (–log10 molar) |
| `pEC50_ci.lower` / `pEC50_ci.upper` | 95% CI bounds — not a target, describes uncertainty |
| `pEC50_std.error` | — |
| `Emax_estimate (log2FC vs. baseline)` | secondary endpoint, orthogonal to potency |
| `Emax_ci.lower` / `Emax_ci.upper`, `Emax_std.error` | uncertainty for Emax |
| `Emax.vs.pos.ctrl_estimate (dimensionless)` | alternate Emax normalization |
| `Emax.vs.pos.ctrl_ci.*`, `Emax.vs.pos.ctrl_std.error` | uncertainty for the above |
| `Split` | `Train` in this file (all rows) — not needed once train/test are separate files |

The unblinded test files should share this same schema (same as train, just with real
`pEC50`/`Emax` values now filled in rather than blank) — verify column names match
exactly once loaded, since the *blinded* test file only has `Molecule Name` +
`CXSMILES (CDD Compatible)` (note: differently-named SMILES column in the blinded file
— rename to `SMILES` if you end up needing it, though for our purposes the unblinded
phase files should already carry a proper `SMILES` column matching train).

Two usable regression targets here: **`pEC50`** (primary, well-populated — 4140/5326
compounds... actually 4140 out of the train file per the notebook) and **`Emax`** /
`Emax_vs_ctrl` (secondary, decide if worth a second dataset config — the tutorial
treats Emax purely as an analysis/QC signal, not a modeling target, so default to just
`pEC50` unless there's a reason to add Emax as its own task).

## 3. Cleaning / filtering steps

1. Concatenate train + (phase1 + phase2 unblinded test), select
   `["Molecule Name", "SMILES", "pEC50"]`.
2. Drop rows with `pEC50` NaN (per the notebook, this drops the training set from 5326
   → 4140 rows; check the analogous drop rate on the test side).
3. Canonicalize SMILES per this repo's usual `Chem.CanonSmiles` step.
4. Preserve train vs. test membership from the source files (don't re-split) — per the
   tutorial, the test set was deliberately constructed via similarity search (ECFP4
   Tanimoto > 0.4 against confirmed actives), so it is **not** a random or scaffold
   split; it's biased toward being structurally close to the training actives. This is
   worth flagging explicitly for whoever is redoing leakage/statistical analysis, since
   it's a different (and more favorable-to-models) train/test relationship than a
   scaffold split would give.

No log-transform needed — `pEC50` is already on a log scale by construction (unlike
ExpansionRx/Polaris raw-unit endpoints).

## 4. Counter-assay filtering (optional, not applied by the tutorial's baseline)

The counter-assay file lets you flag likely non-specific/false-positive hits: a
compound is "potent and selective" only if primary `pEC50 >= 6` **and**
`primary_pEC50 - counter_pEC50 >= 1.5`. The tutorial computes this only for exploratory
plotting, not as a filter on the training data. Decide separately whether the benchmark
should exclude non-selective compounds or leave the task as "raw PXR pEC50 regardless
of mechanism" — this changes what's being measured, so it's a modeling-scope decision,
not a data-cleaning correctness fix.

## 5. Integration point in this repo

- New `source.name` (e.g. `OPENADMET_PXR`) branch in `src/common/data_v2.py::load()`.
- Solver concatenates train + both unblinded test-phase CSVs, builds `splits` from
  original file membership (train file rows → `train`, phase1+phase2 rows → `test`; no
  `valid`).
- Single `config/dataset/reg_pxr_pEC50.yaml`, `task: regression` (plus optionally a
  second `reg_pxr_Emax.yaml` if Emax is wanted as its own task — see §2).
- If Emax is added, apply the same drop-NaN-per-target pattern as the other two
  challenges rather than assuming full row coverage.
