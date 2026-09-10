# Polaris × ASAP Discovery × OpenADMET Antiviral Challenge — Data Guide

> **Status:** downloaded and cleaned by `scripts/prepare_openadmet_data.py`. Raw pulls
> in `data/raw/polaris/`, per-endpoint cleaned CSVs in `data/processed/polaris_admet/`
> and `data/processed/polaris_potency/`. Everything below marked "confirmed" was
> verified against the actual API response, not just the tutorial docs.

Source tutorial reviewed: `data_repos/asap-polaris-blind-challenge-examples`
(`01. Ligand ADMET and Potency (Property Prediction).ipynb`, `evaluation/`).

This repo's notebook demonstrates the **live blind-competition** workflow
(`polaris.load_competition(...)`), where the test set has no labels and you submit
predictions to the Hub for scoring. That competition (both the ADMET and Potency
sub-challenges) **ended in early 2025**, and OpenADMET released fully-labeled
"unblinded" versions of both datasets on 2025-03-28. Use those instead — no blind
submission or competition API needed for our purposes.

## 1. Data source

Two separate sub-challenges, each its own Polaris **dataset** (not competition)
resource:

```python
import polaris as po

admet_ds   = po.load_dataset("asap-discovery/antiviral-admet-2025-unblinded")
potency_ds = po.load_dataset("asap-discovery/antiviral-potency-2025-unblinded")
```

**Package name pitfall (confirmed):** the tutorial's `env.yml` installs `polaris` from
conda-forge, but on PyPI `polaris` is an unrelated project ("Dashboard made easy").
The actual Polaris Hub client is **`polaris-lib`** on PyPI (still `import polaris`).
`pip install polaris` will silently give you the wrong package. Pinned in
`requirements.txt` as `polaris-lib==0.13.0`.

**Auth (confirmed):** `po.load_dataset(...)` on these two unblinded datasets works
with no login/token — they're public.

**Fast bulk access (confirmed, important):** don't iterate rows via
`dataset.rows`/`dataset.get_data(row=..., col=...)` — that issues what looks like a
remote call per cell and is impractically slow (thousands of calls for a 560-row
table). Go through the zarr store directly and vectorize:

```python
ds = po.load_dataset("asap-discovery/antiviral-admet-2025-unblinded")
zr = ds.zarr_root
df = pd.DataFrame({k: zr[k][:] for k in zr.keys()})  # full table in ~1s
```

This is what `scripts/prepare_openadmet_data.py::load_polaris_dataset()` does.

## 2. Schema

**ADMET dataset** (`antiviral-admet-2025-unblinded`, 560 rows total — confirmed
train=434 / test=126):

| Column | Notes |
|---|---|
| `Molecule Name` | compound ID |
| `CXSMILES` | see stereo caveat below |
| `Set` | `Train` / `Test` (confirmed: capitalized) |
| `MLM` | mouse liver microsomes clearance |
| `HLM` | human liver microsomes clearance |
| `KSOL` | solubility, µM |
| `LogD` | already log-scale |
| `MDR1-MDCKII` | permeability, ×10⁻⁶ cm/s |

**Potency dataset** (`antiviral-potency-2025-unblinded`, 1328 rows total — confirmed
train=1031 / test=297):

| Column | Notes |
|---|---|
| `Molecule Name` | compound ID |
| `CXSMILES` | see stereo caveat below |
| `Set` | `Train` / `Test` (confirmed: capitalized) |
| `pIC50 (SARS-CoV-2 Mpro)` | –log10(IC50) |
| `pIC50 (MERS-CoV Mpro)` | –log10(IC50) |

Both are sparse multi-target tables (the tutorial's baseline trains one model per
target rather than jointly). Same approach as ExpansionRx: **treat each of the 7
endpoints (5 ADMET + 2 potency) as its own single-task regression dataset config.**

## 3. CXSMILES stereo caveat

Molecules are given as **CXSMILES**, not plain SMILES — they carry ChemAxon "enhanced
stereochemistry" group annotations, e.g.:
```
COC1=CC=CC(Cl)=C1NC(=O)N1CCC[C@H](C(N)=O)C1 |a:16|
CC(C)[C@H]1C2=C(CCN1C(=O)CC1=CN=CC3=CC=CC=C13)SC=C2 |o1:3|
```
`|a:16|` marks an absolute stereocenter, `|o1:3|` marks an "or" stereo group (racemic /
unassigned relative stereo at that group). This repo's existing `build_dataset()`
canonicalizes SMILES via `Chem.CanonSmiles`, which will parse the CXSMILES fine but
**drops the enhanced-stereo group annotation** in the plain canonical output — i.e. an
"or"-group compound (uncertain absolute configuration) becomes a single fixed
stereoisomer's SMILES with no indication the config was actually ambiguous. This is
probably acceptable for embedding models (none of the wrappers in `model_wrappers/`
appear to consume CXSMILES stereo groups), but flag it explicitly rather than silently
losing information — worth a one-line note in whatever solver you write.

Also note: potency dataset has 5 known duplicate CXSMILES pairs in the test set
(enantiomers that canonicalize to the same 2D structure but had different assay
readouts) — these were filtered by OpenADMET already in the unblinded release per
their dataset description; verify after loading that `Molecule Name` is still unique
per row.

## 4. Known bad datapoints (exclusions)

The tutorial's own evaluation code (`evaluation/utils.py::mask_flagged`) excludes a
small number of flagged-bad measurements before scoring, loaded from
`evaluation/data/exclusions/{admet,potency}.json` (already present in this cloned
repo). Counts:

- ADMET: HLM (4 excluded), MLM (5 excluded), KSOL/LogD/MDR1-MDCKII (0 excluded)
- Potency: pIC50 SARS-CoV-2 Mpro (5 excluded), pIC50 MERS-CoV Mpro (5 excluded)

Each entry has an `idx` (position in the *original blinded competition test array*,
not the unblinded dataset — don't rely on it) and a `CXSMILES` string. **Match by
CXSMILES (or its canonicalized SMILES) rather than by idx** when applying these
exclusions to the unblinded dataset, since row order isn't guaranteed to match. This is
a small, cheap step worth carrying over since these are known-erroneous measurements
the challenge organizers themselves flagged.

**Confirmed gotcha — restrict the match to `Set == "Test"`.** Canonicalizing away the
CXSMILES enhanced-stereo groups (§3) means a flagged compound's plain SMILES can
collide with an *unrelated* train-set row — a different `Molecule Name`, a different
stereoisomer/batch, and a different measured value, not the flagged datapoint at all.
Concretely: matching on canonical SMILES alone against the potency dataset dropped 12
and 11 rows for the two pIC50 targets instead of the expected 5 and 5 — three of the
five flagged compounds per target had a same-canonical-SMILES sibling row sitting in
`Train` (e.g. flagged idx 188 canonicalizes to a SMILES shared by 3 rows: two `Train`
rows with different `Molecule Name`/stereo-group tag and different pIC50 values, and
exactly one `Test` row — only that one `Test` row is the actual flagged exclusion).
Since these exclusions only ever applied to test-set evaluation in the original
competition (`mask_flagged` is only called on the test-set `y_true`/`y_pred` arrays),
restricting the drop to `Set == "Test"` rows recovers the exact official counts (5/5).
`scripts/prepare_openadmet_data.py::fetch_polaris()` does this
(`mask = df["_smiles_canon"].isin(bad) & (df["Set"] == "Test")`) — don't drop on
canonical-SMILES match alone.

## 5. Metric conventions from the official evaluation code

Not required for loading, but useful context if the statistical-testing re-evaluation
work wants to stay comparable to the official leaderboard: `evaluation/admet.py` /
`evaluation/potency.py` log10(x+1)-transform every ADMET target except `LogD` before
scoring (`clip_and_log_transform` in `evaluation/utils.py`), and rank submissions by
mean absolute error with 1000-sample bootstrapping for CIs and a compact-letter-display
significance grouping (`evaluation/cld.py`, via `pairwise_tukeyhsd`).

## 6. Integration point in this repo

- New `source.name` (e.g. `POLARIS`) branch in `src/common/data_v2.py::load()`, using
  `po.load_dataset(name)` where `name` comes from the dataset config (so both
  `antiviral-admet-2025-unblinded` and `antiviral-potency-2025-unblinded` share one
  solver function).
- Split on the `Set` column to build the `splits` dict (`train`/`test`; no `valid`
  provided, same as OGB/ExpansionRx).
- 7 new `config/dataset/reg_polaris_*.yaml` files, `task: regression`.
- Apply the CXSMILES→canonical-SMILES conversion and the exclusions-matching step
  inside this solver, not downstream, since they're specific to how this source's raw
  data is shaped.
