# TabICL / TabPFN scaffold: tweaks and tests

Brief for a coding agent working in the ml-embedding-benchmark repo.

Goal: TabICL and TabPFN are being used as regression heads on frozen chemical
embeddings for OpenADMET endpoints, and their measured performance is below
expectation. This document lists the changes to make and the experiments to run,
ordered by expected value.

Do not run the full ~2900-container fan-out for any of this. Every experiment
below runs on a 3-endpoint x 3-embedding subset. Details in "Experiment harness".

---

## Repo map (files in scope)

| File | Role |
| --- | --- |
| `scripts/compute_prediction_errors.py` | Loads embeddings + prepared data, slices splits, calls `fit_regressor`, writes per-compound errors |
| `src/eval/supervised/regression.py` | Dispatches on `method`; grid-searches sklearn heads, delegates TFM heads |
| `src/eval/supervised/tabicl_head.py` | TabICL fit path |
| `src/eval/supervised/tabpfn_head.py` | TabPFN fit path (not yet reviewed, assume same issues) |
| `src/eval/supervised/models.py` | `get_reg_models(dtype)` -> rf/ridge/knn pipelines and grids (not yet reviewed) |
| `src/eval/supervised/const.py` | `CV_SPLITS` |
| `scripts/modal_score_errors.py` | Modal fan-out |

---

## Ground rules

These are project constraints, not suggestions.

- **No chemical structures in logs.** SMILES, InChI, and mol blocks must never
  reach log output, exception messages, or telemetry. Use `molecule_id` or a
  row index instead. This applies to anything printed by
  `modal_score_errors.py`'s `print(result.stdout)` / `print(result.stderr)`
  path, which currently forwards all subprocess output including any RDKit
  sanitization warnings.
- **Dependencies.** Use `uv` with a committed `uv.lock` and `exclude-newer` set
  to a rolling 3 days back. `scipy`, `pyyaml`, and `torch` are currently
  unpinned across all four Modal image builders in `modal_score_errors.py`.
  Pin them. Unpinned `torch` across the tabicl and tabpfn images is also a
  benchmark confound, since the two images can resolve to different versions
  depending on when each layer was last built.
- **Code comments.** Only for non-obvious choices or context that prevents
  future breakage. Anything explanatory belongs in the PR description.
- **Run `pytest` and `pre-commit` before committing.**

---

## Reference facts about TabICL

Sourced from the TabICLv2 docs and paper (arXiv 2602.11139) and from
Ben Hicham et al., "Tabular foundation models for in-context prediction of
molecular properties" (arXiv 2604.16123). Do not re-derive these from memory.

- TabICLv2 is pretrained on datasets with **2 to 100 columns** and 300 to 48K
  training samples. It generalizes past that, with reported good performance up
  to ~2000 features, but the current inputs (2048 to 3072 dims) sit at the far
  edge.
- Runtime is **O(n² + nm²)** in rows and features. Feature width is the
  dominant term at current dimensionalities.
- `fit()` stores data and loads the checkpoint. **All computation happens in
  `predict()`.** Any code path that triggers extra predicts pays full cost.
- `batch_size` controls how many ensemble members are resident at once. It does
  **not** change ensemble size. `n_estimators` (default 8) sets ensemble size
  and costs linear wall-clock at fixed `batch_size`, with no change to peak
  memory.
- `TabICLRegressor` already standardizes `y` internally via `y_scaler_` and
  inverts on predict. **Do not standardize the target manually.**
- `norm_methods` defaults to `["none", "power"]`. Other options:
  `"quantile"`, `"quantile_rtdl"`, `"robust"`.
- `outlier_threshold` defaults to `4.0` (z-score clipping).
- `predict(X, output_type=...)` accepts `"mean"` (default), `"median"`,
  `"quantiles"`, `"raw_quantiles"`.
- `use_amp="auto"` enables mixed precision once n >= 1024 **or** features >= 60.
  At current widths it is always on.
- `random_state` defaults to `42`, so TabICL is deterministic across runs unless
  changed. Repeated "seeds" produce identical predictions.
- `softmax_temperature`, `class_shuffle_method`, and `average_logits` are
  classifier-only. They do not exist on `TabICLRegressor`.
- The preprocessing recipe behind published TFM-on-molecules results:
  fit the scaler on the training portion only, standardize, clamp to
  **[-6, 6]**, and leave the model's internal normalization in place.
- Representation quality dominates predictor choice in every published
  comparison. Descriptor-derived and descriptor-pretrained representations
  (CheMeleon, RDKit2d, Mordred) beat Morgan/ECFP substantially; SMILES
  language-model embeddings (SMI-TED class) rank near the bottom.

---

## Phase 0: correctness guards

Land these first. They are cheap and they determine whether any later numbers
are trustworthy.

### 0.1 Assert the two index spaces agree

`compute_prediction_errors.py::compute_errors` mixes two coordinate systems.
`surviving_positions` (from `embedded.y.index`) indexes the original prepared
dataset. `train_idx` / `test_idx` (from `embedded.splits`) index the surviving
rows of `X`. These agree only if `remove_failed_embeddings()` remaps splits when
it drops rows.

Add after the split slicing:

```python
all_idx = sorted(train_idx + test_idx)
assert all_idx == list(range(len(X))), (
    f"splits do not partition surviving rows: "
    f"{len(all_idx)} split indices over {len(X)} rows"
)
```

Then verify `remove_failed_embeddings()` in `src/common/types.py` actually
remaps. Report what you find. Check the embeddings known to have had trouble
first: `ChemFM-1B`, `ChemFM-3B`, `grover_large`, `grover_base`, `rmat_4M`.

### 0.2 Fail loudly when nothing is produced

`compute_errors` returns `None` on three paths (load failure, missing file,
row-count mismatch). `main()` does `continue` and exits 0, so
`modal_score_errors.py` reports `ok` for containers that wrote no file.

Track produced-vs-skipped counts in `main()` and exit non-zero when zero
endpoints produced output.

### 0.3 Make result writes atomic

`joblib.dump(result, output_path)` is not atomic. Combined with the
resume-by-existence check (`if os.path.exists(output_path) and not
args.override`), a preempted or timed-out container can leave a truncated file
that a resumed run then skips as complete.

Dump to `output_path + ".tmp"` and `os.replace()` into position.

Then audit the existing Volume for already-corrupt artifacts: load every
`data/errors/**/*.joblib` and check that `molecule_id`, `y_true`, and `y_pred`
are equal-length and finite.

### 0.4 Add input validation to the TFM heads

`X` currently reaches TabICL unchecked. TabICL imputes NaN natively while
sklearn heads raise, so the same input behaves differently per method and the
comparison is silently inconsistent.

In `fit_tabicl_regressor` and `fit_tabpfn_regressor`, assert `np.isfinite(X).all()`
and `np.isfinite(y).all()`, and cast `X` to `float32`.

---

## Phase 1: the preprocessing asymmetry

**This is the highest-value change. Do it before any tuning.**

### 1.1 Determine what the baselines get

In `regression.py::fit_regressor`, the `tabicl` and `tabpfn` branches `return`
before `get_reg_models(X.dtype)` is called. So any preprocessing built into
those sklearn pipelines applies to `rf`/`ridge`/`knn` only.

Read `src/eval/supervised/models.py` and report, for each of rf, ridge, knn:
- whether the pipeline contains a `StandardScaler` or any other transformer
- what `X.dtype` changes about the returned models and grids

If ridge and knn are scaled and TabICL is not, the published comparison is
between scaled baselines and an unscaled TFM. Ridge and kNN are strongly
scale-sensitive. Flag this in the PR regardless of the outcome.

### 1.2 Move the TFM heads onto a Pipeline

The scaler has to live inside the returned estimator, because
`compute_prediction_errors.py` calls `fit_result["model_obj"].predict(X_test)`
with an untransformed `X_test`. A `Pipeline` handles that automatically.

In `tabicl_head.py`:

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from tabicl import TabICLRegressor

_BATCH_SIZE = 1
_CLAMP = 6.0


def _clamp(X, limit):
    return np.clip(X, -limit, limit)


def _build(**tabicl_kwargs) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clamp", FunctionTransformer(_clamp, kw_args={"limit": _CLAMP})),
        ("model", TabICLRegressor(batch_size=_BATCH_SIZE, **tabicl_kwargs)),
    ])
```

Use a module-level `_clamp` rather than a lambda so the estimator stays
picklable if it is ever persisted.

Note that `StandardScaler` fits on `X_train` only, which is what you want, and
that this leaves TabICL's internal normalization untouched, per the published
recipe.

Apply the same change to `tabpfn_head.py`.

### 1.3 Test

Config A (current, no scaling) vs Config B (scale + clamp). Run the matrix from
"Experiment harness" and report Spearman deltas per cell.

---

## Phase 2: estimator configuration

Each of these is a single-parameter change. Run them as independent A/Bs against
whichever of Config A or B wins Phase 1.

### 2.1 `n_estimators`

Default is 8. Peak memory is bounded by `batch_size=1`, so raising this costs
wall-clock only. The TabICL authors explicitly recommend raising it in
high-dimensional feature regimes, which is exactly the current setting.

Test `n_estimators` in `{8, 16, 32}`.

### 2.2 `output_type="median"`

`predict()` returns the mean of the predicted distribution by default. Every
OpenADMET endpoint here is scored with Spearman, and TabICL is quantile-based
internally, so the median frequently ranks better on skewed conditionals.

`TabICLRegressor` takes `output_type` at predict time, not at construction, and
`Pipeline.predict(X)` is called without kwargs. Subclass to change the default:

```python
class _MedianTabICLRegressor(TabICLRegressor):
    def predict(self, X, output_type="median", alphas=None):
        return super().predict(X, output_type=output_type, alphas=alphas)
```

The subclass inherits `__init__`, so `sklearn.base.clone` and `get_params` keep
working.

### 2.3 `norm_methods`

Default `["none", "power"]`. Transformer-derived embeddings carry high-magnitude
rogue dimensions that Yeo-Johnson handles poorly.

Test `"quantile"` and `"robust"`.

### 2.4 `outlier_threshold`

Default `4.0`. Those same rogue dimensions are what z > 4 clipping flattens.
Note this interacts with the [-6, 6] clamp from Phase 1.

Test `{4.0, 8.0}`.

### 2.5 `use_amp=False`

Always on at current widths under the auto heuristic. Unstandardized wide
embeddings under fp16 autocast is a plausible precision loss; this becomes less
likely once Phase 1 lands, so run it after.

Test on one endpoint only unless it shows an effect.

---

## Phase 3: dimensionality reduction

Current inputs are 2048 to 3072 dims against a model pretrained on 2 to 100
columns, with runtime quadratic in feature count. Reduction targets accuracy,
memory, and wall-clock simultaneously.

Insert PCA into the Phase 1 pipeline between the clamp and the model:

```python
Pipeline([
    ("scale", StandardScaler()),
    ("clamp", FunctionTransformer(_clamp, kw_args={"limit": _CLAMP})),
    ("pca", PCA(n_components=256, random_state=0)),
    ("model", TabICLRegressor(batch_size=_BATCH_SIZE)),
])
```

Test `n_components` in `{full (no PCA), 512, 256, 128}`.

`n_components` must not exceed `min(n_train_rows, n_features)`. At ~5000 train
rows all four settings are valid, but guard it for smaller endpoints:
`n_components=min(k, X.shape[0] - 1, X.shape[1])`.

Expected outcome: `nm²` drops by 16x at 512 and 256x at 128. If Spearman holds,
this resolves the OOM, the ~100-minute worst-case fit, and the GPU tier
requirement in one change.

Report wall-clock and peak GPU memory alongside Spearman for each setting.

---

## Phase 4: remove wasted compute

### 4.1 Delete or gate the `cross_val_score` block

`fit_tabicl_regressor` runs `cross_val_score(..., cv=CV_SPLITS)` before the
final fit. Since TabICL does all its work at predict time, this is 5 additional
full O(n² + nm²) passes to produce a `cv_score` that nothing selects on
(`best_params` is `{}`). Six passes per fit total. This is the likeliest source
of the long tail in fit times.

The number is also not comparable to the sklearn heads' `cv_score`, which comes
from `GridSearchCV.best_score_`. And because in-context learning scales with
context size, a fold holding out 20% systematically understates what the
final full-context model does.

Remove it, or put it behind a `--with-cv-score` flag defaulting off. Get
method comparisons from held-out test predictions instead.

Check `tabpfn_head.py` for the same pattern.

### 4.2 Fix the CV splitter

`cv=CV_SPLITS` as an integer gives a regressor `KFold(shuffle=False)`. If
`data/prepared` retains any assay, series, or source ordering, folds are grouped
by whatever that ordering encodes.

This matters more in `regression.py::fit_regressor`, where the same integer
feeds `GridSearchCV` and therefore drives actual hyperparameter selection for
rf, ridge, and knn.

Replace with an explicit splitter in both places:

```python
from sklearn.model_selection import KFold
cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=0)
```

Then check whether `data/prepared` is ordered, and report.

---

## Phase 5: representation

Lower priority for the agent, but it likely bounds everything above.

Published benchmarks consistently rank descriptor-based and descriptor-pretrained
representations well above ECFP/Morgan when paired with tabular foundation
models, and rank SMILES language-model embeddings below ECFP.

- Check whether the ~57 embedding stems already include RDKit2d descriptors,
  Mordred descriptors, or CheMeleon. If not, adding RDKit2d (217 dims) is cheap
  via `scikit-fingerprints` and is a strong reference point. At 217 dims it also
  lands inside TabICL's pretrained column range without PCA.
- Report which stems are SMILES-LM derived, so the results table can be read
  with that grouping in mind.

---

## Experiment harness

### Subset

Three endpoints spanning size, and three embeddings spanning representation
family. Use the existing per-endpoint fan-out grain.

```
--endpoints  reg_openadmet_expansionrx_LogD, reg_openadmet_pxr_pEC50, <one small endpoint>
--embeddings ECFP, <one SMILES-LM stem>, <one descriptor or graph stem>
--methods    tabicl
```

Include `ridge` as a fixed reference in every run. It is fast, scale-sensitive,
and will make the Phase 1 result unambiguous.

### Protocol

- One variable per run. Do not stack changes before Phase 1 resolves.
- Write results to a separate output root (`data/errors_exp/<config_name>/`) so
  the production `data/errors` tree stays clean.
- TabICL is deterministic at `random_state=42`, so single runs are sufficient
  for TabICL. Vary the split seed, not the model seed, if you need error bars.
- Record per cell: test Spearman, wall-clock seconds, peak GPU memory.

### Reporting

One markdown table per phase, rows = (endpoint, embedding), columns = configs,
values = test Spearman with the delta against the phase baseline. Plus a short
paragraph stating which config wins and whether the win is consistent across
representation families.

---

## Acceptance criteria

- [ ] Phase 0 guards landed, with a report on whether `remove_failed_embeddings`
      remaps splits and whether any existing `data/errors` artifacts are corrupt
- [ ] `models.py` audited; the preprocessing asymmetry between sklearn heads and
      TFM heads documented explicitly
- [ ] Phase 1 A/B complete, with the ridge reference included
- [ ] Phase 2 sweeps complete against the Phase 1 winner
- [ ] Phase 3 PCA sweep complete, reporting accuracy, time, and memory
- [ ] `cross_val_score` removed or gated; measured speedup reported
- [ ] CV splitter made explicit in both `tabicl_head.py` and `regression.py`
- [ ] Modal image dependencies pinned via `uv`
- [ ] `pytest` and `pre-commit` pass
- [ ] Summary recommending a single production configuration

---

## Open questions to answer in the PR

1. Does `remove_failed_embeddings()` remap `embedded.splits`?
2. Do the rf/ridge/knn pipelines in `models.py` include a scaler?
3. What does `get_reg_models(X.dtype)` branch on, and does it change grids?
4. Is `data/prepared` row order meaningful?
5. Does `tabpfn_head.py` share the `cross_val_score` and no-preprocessing
   patterns found in `tabicl_head.py`?
6. Which of the ~57 embedding stems are descriptor-based vs SMILES-LM derived?
