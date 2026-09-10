# OpenADMET migration + embedding extraction — status

Context for future sessions picking this up. Written 2026-08-05.

## Why this work exists

This benchmark originally scored embedding models on MoleculeNet/TDC tasks. The
top-performing model in the original paper likely saw the MoleculeNet test
molecules during its own pretraining — a data leakage problem that inflates
its apparent performance. The fix in progress: replace those tasks with three
OpenADMET challenges (ExpansionRx, Polaris antiviral ADMET/Potency, PXR) that
postdate every embedding model's pretraining cutoff.

Scoring/statistical-analysis changes are explicitly **out of scope** for this
pass — the current goal is just getting clean embeddings computed and cached
for all 17 new endpoints across as many of the 15 model wrappers as feasible.

## What's done

1. **Data sourcing** — `docs/data_sources/{expansionrx,polaris_antiviral,pxr}.md`
   document exactly which HF/Polaris repos to pull from (all three challenges
   have concluded and unblinded test sets exist — the tutorials themselves
   only show the live-challenge blind-submission flow, not this).

2. **Extraction** — `scripts/prepare_openadmet_data.py` downloads and cleans
   all three sources into `data/processed/<source>/<endpoint>.csv`
   (`molecule_id, smiles, value, split` columns). Produces 17 endpoints: 9
   ExpansionRx (log1p-style transform applied per the tutorial's own
   convention) + 5 Polaris ADMET + 2 Polaris Potency (officially-flagged bad
   datapoints dropped, matched by canonical SMILES restricted to `Set ==
   "Test"` rows — matching by canonical SMILES alone over-drops unrelated
   train rows that collide after stripping CXSMILES stereo annotations) + 1
   PXR pEC50.

3. **Pipeline integration**:
   - `src/common/data_v2.py::load()` — new `LOCAL_CSV` source branch reads
     those processed CSVs directly (no graph column needed; confirmed via
     `grep -rl GraphEmbedder model_wrappers/*/wrapper.py` that every current
     wrapper is a `SmilesEmbedder`, none consume `dataset.data['graph']`).
   - `config/dataset/reg_openadmet_*.yaml` — 17 new dataset configs, `task:
     regression`, shared `reg_openadmet_` prefix for glob targeting.
   - `data/prepared/reg_openadmet_*.joblib` — already materialized locally via
     `python download.py --multirun '+dataset=glob(reg_openadmet_*)'`.
   - Fixed `src/common/types.py::Dataset.labels` — was unconditionally
     dropping a `graph` column that doesn't exist for `LOCAL_CSV`-sourced
     datasets (`errors='ignore'` added to the `.drop()` call).

4. **Repo-level fixes** made along the way (real bugs, not Modal-specific
   workarounds — they help local runs too):
   - Pinned previously-unpinned git dependencies to commit SHAs:
     `clamp_repo/setup.py` (`mhnreact`, `openai/CLIP`), `coati_repo/setup.py`
     (`due`), `model_wrappers/{graphmvp,huggingmolecules}/init.sh`
     (`bp-kelley/descriptastorus`), `model_wrappers/mol2vec/init.sh`
     (`samoturk/mol2vec`).
   - Removed a dead `import pkg_resources` from `clamp_repo/setup.py` (was
     causing a confusing recursive PEP517 build failure under modern
     setuptools, which no longer vendors `pkg_resources`).
   - Pinned `model_wrappers/huggingface/init.sh`'s `transformers` install to
     `==4.46.3` — unpinned, it drifts to a version that removed
     `transformers.modeling_utils.SequenceSummary`, which `wrapper.py` imports
     directly.
   - Added a `.python-version` file (`3.10.15`... locally; `3.11` on Modal) to
     `model_wrappers/huggingface/`, which had none.

5. **Modal harness** — `scripts/modal_embed.py`. Builds one Modal Image per
   wrapper (running that wrapper's own `init.sh` at image-build time instead
   of per-container-start), fans out via `.map()`-free `spawn()`/`get()` since
   each wrapper needs a distinct image. Key findings baked into it:
   - Modal's client library hard-requires Python ≥3.10 in the *container's
     own* interpreter (not just an image-builder setting) — confirmed live.
     `graphfp`, `graphmvp`, `mol-r` (cp37-specific wheel URLs) and `molbert`
     (`torch==1.4.0`/`transformers==3.5.1`, no wheels for ≥3.10) are
     permanently blocked there until someone builds a dual-interpreter image
     (modern Python for Modal's own runtime + the pinned legacy Python
     installed via pyenv, invoked via subprocess).
   - `cddd`, `chemformer`, `simson`, and `huggingface`'s `selformer`/
     `selformer_lite` variants are excluded — need interactive Google
     Drive/Box downloads a container can't do non-interactively.
   - **One wrapper's image-build failure aborts the entire batch** — Modal
     needs to register/build every requested function before running any of
     them, so `--wrappers a,b,c` where `a` fails to build means `b` and `c`
     never even get attempted. Run wrappers one at a time (or in small batches
     you're confident about) rather than lumping everything into one call.
   - `modal app logs <app-id>` has been unreliable this session (hits a
     "Logs query hit resource limit" error even with `--tail`/`--search`
     filters, for reasons not yet understood — possibly related to log volume
     from very verbose pip output). When this happens, check the Volume
     directly (`modal volume ls ml-embedding-benchmark /embedded/<dataset>`)
     as ground truth for whether a run actually produced anything, rather
     than relying on logs.
   - Non-detached `modal run` invocations tear down the whole ephemeral App
     when the local client disconnects — including from a `timeout`-induced
     kill. This looked like "crash looping" before that was understood. Use
     `modal run --detach` for anything that might outlive a short local wait.

6. **Pivoted to local execution** for the wrappers Modal batch-building kept
   choking on. Set up `pyenv` (`~/.pyenv`) with Python 3.8.18/3.9.19/3.10.15
   built from source (needed build-deps via `apt-get install build-essential
   libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm
   libncursesw5-dev tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
   git` — if pyenv builds fail with `No module named 'zlib'` or similar,
   check these actually finished installing, not just started). Confirmed
   `pyenv shell 3.9` resolves bare major.minor to the latest installed patch
   automatically — the existing wrapper `.python-version` files (bare
   `"3.9"` etc.) work as-is, no edits needed.

   Local run script: `/tmp/claude-1000/.../scratchpad/run_wrapper_local.sh`
   (session-scoped scratchpad path, **will not persist** — recreate from the
   pattern in this doc's "Local run recipe" section below if it's gone).

## Current state of each wrapper

| Wrapper | Status | Notes |
|---|---|---|
| `fingerprints` | ✅ done, 17/17 | Ran on Modal originally; no legacy deps, simplest wrapper. |
| `clamp` | ✅ done, 17/17 | Local. Needed the `pkg_resources` dead-import fix. |
| `coati` | ✅ done, 17/17 | Local. Needed the `due` git-SHA pin + `rdkit` in base install. |
| `mol2vec` | ✅ done, 17/17 | Local. First attempt dropped 1/17 (`LogD`) with a `pickle data was truncated` error from the *first* job in the Hydra sweep — re-running (cache-aware) picked up just the gap. Seen the same "first job in a batch is flaky" pattern on Modal too (fingerprints); root cause not fully understood, but re-running fixes it. |
| `huggingmolecules` | 🔄 running locally, fixed | **Was actually dead, not "running"** — `/tmp/local_run_huggingmolecules.log` shows it crashed 2026-08-05 ~20:44 with only ~12/102 (17 datasets × 6 model variants) combos done. Two real bugs in `GroverEmbedder` (`grover_base`/`grover_large`), both only surfacing once real data hit them: (1) a genuine `torch.OutOfMemoryError` on this machine's actual T4 GPU (confirmed present via `nvidia-smi` — this wrapper's torch build can use it, unlike `huggingface`'s) at the base `batch_size=128`, memory use driven more by which molecules land in a batch than batch size itself (empirically, `bs=64` peaked *higher* than `bs=128` depending on content) — dropped to `batch_size=16`, empirically stable with far lower peak memory; (2) the OOM should have triggered the per-molecule fallback retry, but `GroverEmbedder` never overrode `process_step()`, so the base class's `self._model(batch)` call returned Grover's raw `(atom_ffn_output, bond_ffn_output)` tuple instead of a concatenated Tensor, and `torch.cat` in `safe_batch()` crashed with `TypeError: expected Tensor as element 0 ... but got tuple` — turning one bad batch into a fatal crash for the whole 102-job sweep. Also, `GroverEmbedder` never set `OUTPUT_DIM` at all (needed by the NaN-fallback path), and the concatenated dim differs by variant (1600 for base, 2400 for large) so it's now measured once via a dummy forward pass rather than hardcoded. Verified: both variants produce correct-shaped output and gracefully NaN-out genuinely-invalid SMILES instead of crashing. Full 17-dataset run restarted. |
| `unimol` | 🔄 running on Modal, fixed | **Not actually still running locally** — the previous session's local process (`/tmp/local_run_unimol.log`) died 2026-08-05 17:17 on its very first job with 0/17 datasets ever completed. Real bug, not slowness: `wrapper.py` did dict-style `unimol_repr["cls_repr"]`, but the installed `unimol_tools` version's `get_repr(..., return_atomic_reprs=False)` always returns a plain list (confirmed by reading its source and calling it directly) — every batch hit this, fell through to the per-molecule retry path, which has the exact same bug, then crashed a second time in the NaN-fallback path because `self._output_dim` was still `None` (never set, since no batch had ever succeeded). Fixed all three (see `wrapper.py`). Also found `use_cuda=False` was hardcoded for no real reason — `unimol_tools`'s own `use_cuda` handling already no-ops to CPU when CUDA isn't available, so it's now wired to `get_device()` like the rest of the repo, confirmed live using `cuda:0` on Modal. New `scripts/modal_embed_unimol.py`: single-wrapper Modal script (avoids Finding 2 below), one container per dataset for parallelism, T4 GPU by default. Took 3 attempts to get a clean smoke test: (1) numpy ABI crash — `init.sh`'s `pip install unimol_tools --upgrade` pulls numpy past the base `<2.0` pin, same issue the local run recipe already had a workaround for, just missed replicating it here; (2) `ImportError: libXrender.so.1` — `unimol_tools` imports `rdkit.Chem.PandasTools` → `rdkit.Chem.Draw` at import time for no functional reason, needing X11 libs `debian_slim` doesn't ship (`apt_install("libxrender1", "libxext6", "libsm6")` fixed it). Third attempt succeeded end-to-end (confirmed `unimolv1.joblib`/`unimolv2.joblib` on the volume) — full 17-dataset run kicked off, in progress. |
| `gem` | ✅ done, 17/17 on Modal | See "Modal scoping" below — ran cleanly once actually tried against a single dataset. No code changes needed. |
| `huggingface` | ✅ done locally, 17/17 for all 11 non-selformer variants | The long-running local process (`PID 1767182`) finished `ChemBERTa`(×6)/`ChemGPT`(×2)/`ChemFM`(×2) fully on its own (10/11 variants, all 17 datasets) — only `MolFormer` was stuck at 0/17 throughout. Root cause: `wrapper.py` imports `from transformers.modeling_utils import SequenceSummary` at module level — a **dead import**, never actually referenced anywhere else in the file or repo (confirmed via repo-wide grep) — but it's the entire reason `init.sh` pins `transformers==4.46.3`. `MolFormer`'s HF Hub remote code needs `transformers.masking_utils`, which doesn't exist until `5.0.0`; confirmed via version probing (in a scratch venv) that **no transformers version has both** `SequenceSummary` (removed at 4.53.0) and `masking_utils` (added at 5.0.0) — a real, unbridgeable gap, not a bad pin choice. Deleted the dead import (harmless either way) and, rather than bump the shared venv (risk: `embed.py` instantiates every model via `get_embedder()` *before* checking cache, so even cache-hit jobs for the other 10 variants would've re-run their loading code under the new `transformers` — untested, unnecessary risk to already-completed work), built a second venv (`model_wrappers/huggingface/venv_molformer/`, same `.python-version`, `transformers[torch]>=5.0.0`) used only for `MolFormer`. Ran `+model=glob(huggingface_molformer)` through it — 17/17, done, zero risk to the other 10 variants. On **Modal**: ChemFM-1B/3B alone on A100 done, 17/17; the other variants on T4 still fail for an unresolved reason (moot now that local covers everything) — not investigated further. `SELFormer`/`SELFormer-Lite` moved to Modal entirely (see own row below) rather than staying local — see there for why. |
| `selformer`/`selformer-lite` | ✅ done, 17/17 on Modal | Checkpoint download (`SELFormer.zip`/`SELFormer-Lite.zip`, ~320MB/~217MB) worked via the same `gdown` trick as `cddd`/`simson` — the Google Drive folder in `init.sh` contains exactly those two files, no extras. Ran locally first (Python 3.10, same venv as the rest of `huggingface`) and worked correctly, but slowly: CPU-only (same old-driver issue as `huggingface`'s other variants), and `BATCH_MODE=False` means one molecule at a time — ~2.2 molecules/sec, on pace for many more hours across the remaining large datasets. Moved to a new dedicated Modal script, `scripts/modal_embed_selformer.py` (same one-container-per-dataset pattern as `scripts/modal_embed_unimol.py`) rather than adding it back into the main `huggingface` Modal sweep, specifically to avoid that sweep's still-unresolved GPU-memory issue (loads ~10 different models sequentially in one long process) — each `selformer` container here only ever loads its 2 (small) models, once. Smoke-tested on one dataset, confirmed both `.joblib` files land correctly on the Volume; full 17-dataset run in progress. One correctness check worth noting: `_model_step()`'s output has no leading batch dimension (unlike `ChemGPTEmbedder`'s equivalent), so the base class's `torch.cat(outputs)` for `BATCH_MODE=False` models produces a flattened 1D array, not a 2D one — looked like a bug at first, but `EmbeddedDataset.remove_failed_embeddings()` (`src/common/types.py`) already reshapes 1D `X` back to `(n_samples, -1)` before saving, and this path has no partial-failure handling (one bad molecule crashes the whole call rather than dropping a row), so the reshape is always valid. Confirmed by loading a saved `.joblib` directly: correct shape, no NaN, distinct rows. |
| `cddd` | ✅ done, 17/17 | Needs Python **3.6** specifically (built via pyenv, `3.6.15`) — the oldest of any wrapper here. Two dependency snags, both real: (1) bare `rdkit` has never published a cp36 wheel (`rdkit` on PyPI only starts at 2022.3.3, cp37+); the old package name `rdkit-pypi` has cp36 wheels up to `2021.9.4` — but (2) `rdkit-pypi==2021.9.4` requires `numpy>=1.19`, while `tensorflow==1.10.0` (pinned in `requirements.txt`, itself needed for `cddd_repo`'s TF1-era `InferenceModel`) requires `numpy<=1.14.5` — a direct, unbridgeable conflict between the two pins. Resolved by just not installing `rdkit` at all in this venv: `wrapper.py` never imports it, and `embed.py`'s own `from rdkit import RDLogger` is wrapped in a `try/except ImportError: pass`, so it's genuinely optional here (SMILES are already canonicalized once, at the `data/prepared/*.joblib` stage). `torch` (needed transitively by `src/common/types.py`) pinned to `1.10.2`, the last version with a cp36 wheel. |
| `simson` | ✅ done, 17/17 | Checkpoint download (blocked on a Google Drive rate limit last session) succeeded cleanly once retried — the limit had cleared. Needs Python 3.9 (already built). Straightforward otherwise; worked on the first try once weights were in place. |
| `chemformer` | ✅ done, 17/17, all 3 checkpoints | Box download confirmed not scriptable (client-rendered SPA); user downloaded the 3 `.ckpt` files manually and ran `fix_checkpoint.py` (already present in the wrapper dir, just never wired into `init.sh` — its call is commented out) to rename `vocab_size`→`vocabulary_size` in each checkpoint's hyperparameters and unwrap the `module` key left over from deepspeed training. Needs Python 3.9 + `torch==1.13.1`/`pytorch-lightning==1.2.3`/`deepspeed==0.9.0` (builds fine, just slow) + `git+.../pysmilesutils`. Worked on the first try once the venv was up; fast on GPU (~9s/dataset). |
| `graphfp` | ✅ done, 17/17 | pyenv 3.7.17 + venv, `init.sh` ran clean (all cp37 wheels — torch 1.9.0+cpu, torch_scatter/torch_sparse from data.pyg.org — still live). Both `GraphFP-C`/`GraphFP-CP`; checkpoints were already bundled in `graphfp_repo` (no download needed). Saved as `.json` (legacy path, Python 3.7 < 3.9). |
| `mol-r` | ✅ done, 17/17 | Same pyenv 3.7.17 setup. All 4 variants (`gcn`/`gat`/`sage`/`tag`); checkpoints (`model.pt` per variant) were already present — an earlier scan of `saved/` only found `.pkl` files because it didn't match the `.pt` extension, not an actual gap. `openbabel-wheel==3.1.1.3` installs with a "yanked" warning (needs `libboost_iostreams`) but isn't imported anywhere in the featurizer path we use, so harmless here. |
| `molbert` | ✅ done, 17/17 | Same pyenv 3.7.17 setup. Its own `init.sh` already auto-downloads weights non-interactively (`wget` from figshare, 967MB) — no manual step needed. **Real bug found and fixed**: `MolBertFeaturizer.transform()` runs its *entire* input through BERT in one single forward pass with no internal batching, and `wrapper.py`'s `forward()` passed the whole per-dataset SMILES list straight through — OOM-killed the process twice (once alongside other jobs at 15.8GB anon-rss, once running alone with memory freed at 11.8GB) before being traced to this. Fixed by chunking through `src.common.utils.batch()` like every other wrapper does; finished cleanly afterward. |
| `graphmvp` | ✅ done, 17/17, all 14 variants | Checkpoints: `graphmvp_repo`'s README links a Google Drive folder (`1uPsBiQF3bfeCAXSDd4JfyXiTh-qxYfu6`) containing exactly the 8 baseline-strategy weights (`AM`/`Contextual`/`CP`/`EP`/`GPT_TNN`/`GraphCL`/`IG`/`Motif`, ~7.5MB each) plus a `GraphMVP_complate_features_for_regression.zip` with the `GraphMVP_AM`/`GraphMVP_CP` checkpoints our config actually needs — downloaded just those (skipped the classification zip/poster/slides), placed under `model_wrappers/graphmvp/weights/`. **Two real bugs found and fixed**: (1) `wrapper.py`'s `GeneralGNNWrapper` (the 8 baseline-strategy variants) picked `GNNComplete` vs `GNN` architecture based on the *dataset's* task (regression/classification) — but those 8 checkpoints were only ever trained with the simpler `GNN` architecture regardless of downstream task, so every regression dataset hit a state_dict key-mismatch; fixed by always using `GNN` for that wrapper class (`GraphMVPWrapper`, the actual `GraphMVP_AM`/`GraphMVP_CP` strategy, has genuinely separate simple/complete-feature checkpoints per task and was unaffected). (2) `config/experiment/graphmvp.yaml`'s model sweep only globbed `graphmvp_gnn_*` (the 8 baseline configs), silently dropping the 6 `graphmvp_am_*`/`graphmvp_cp_*` configs (the actual GraphMVP method) — a commented-out explicit list right above the glob was the only trace of them ever being included. Widened to `glob(graphmvp_*)`, which matches all 14 configs exactly; re-run picked up just the missing 6 (cache-aware). |

All four needed `PIP_INDEX_URL=https://pypi.org/simple/` to install anything — the org's private CodeArtifact index returned 401s mid-session (likely an expired temp credential), and these ancient pinned versions (`torch==1.4.0`, `scikit-learn==0.21.3`, etc.) wouldn't be in an internal mirror anyway. All pinned versions still resolve on public PyPI/data.pyg.org/figshare as of this writing.

## Modal scoping for `gem`/`huggingface` (2026-08-06 session)

Before committing to a full re-run, added `--dataset-glob` and `--model-glob` CLI
overrides to `scripts/modal_embed.py`'s `local_entrypoint` (default: all 17
datasets / that wrapper's own model glob) so a single dataset+model combo can be
smoke-tested cheaply instead of waiting on all 17 datasets × every model variant.

**Finding 1 — the previous "failed twice, cause undiagnosed" wasn't a code bug.**
A live smoke test (`--dataset-glob 'glob(reg_openadmet_pxr_pEC50)'`, huggingface
further scoped to one model variant) completed cleanly for **both** wrappers on
the first try, no changes needed — confirmed by checking the Volume directly
for `GEM.joblib` / `ChemBERTa-10M-MTR.joblib` under that dataset. Whatever failed
before was environmental/transient (most likely the documented "non-detached
`modal run` gets killed when the local client disconnects" issue compounding
with finding 2 below), not a real dependency/wrapper problem.

**Finding 2 — every invocation rebuilds/verifies every wrapper's image, not just
the ones you asked for.** `_build_embed_functions()` (in `modal_embed.py`) calls
`app.function()` — which triggers an image build — for every wrapper in
`WRAPPERS` minus `_UNFIXED_LEGACY_PYTHON_WRAPPERS` at *module import time*,
regardless of what `--wrappers` you pass to `main()`. In practice this means a
single-wrapper, single-dataset smoke test still pays for building/verifying all
~8 non-excluded images (fingerprints/clamp/coati/gem/huggingface/
huggingmolecules/mol2vec/unimol) serially, which took 15–20+ minutes total in
this session's test run before the actually-requested wrapper's function even
started. This is almost certainly why earlier attempts run with a short
foreground timeout looked like silent failures — they were killed mid-build,
before any real work (or error) had a chance to show up. Not fixed this
session (would mean restructuring `EMBED_FUNCTIONS` to only decorate the
requested subset, which Modal's current API makes awkward since `app.function`
is a decorator applied at definition time) — worth revisiting if iterating on
individual wrappers becomes a frequent need.

**Finding 3 — the full `huggingface` run still failed, separately, on ChemFM.**
After the smoke test passed, a full 17-dataset/all-variant run was launched and
failed partway through (~30% into one dataset's model loop) with only a bare
`RuntimeError: embed.py failed for wrapper=huggingface (exit 1)` — no traceback
visible, because the dense per-molecule `tqdm` progress output (`\r`-updated,
captured as one log line per frame) appears to have flooded/truncated Modal's
captured stdout/stderr before the actual error text. Given `ChemFM-3B` is a
3-billion-parameter causal LM and every huggingface variant is pinned to a T4
(16GB) regardless of size, a CUDA OOM on ChemFM-3B is the leading suspect —
not confirmed via traceback, but consistent with local runs handling ChemFM
fine on CPU (slow, no VRAM ceiling) while T4 struggled. Split the re-run into
ChemFM-1B/3B on `--modal-gpu A100` separately from everything else on T4 to
test this; both halves were still in progress in the Modal build queue as of
this writing (see Finding 2 for why that took a while just to start).

## Checkpoint downloads (2026-08-06 session)

Investigated whether `cddd`/`chemformer`/`simson`'s "need an interactive
Google Drive/Box download" blocker actually requires a human, or just a tool
that knows how to handle Google's/Box's non-file-API share links.

- **`cddd`: solved, fully automatable.** The share link in `init.sh`
  (`drive.usercontent.google.com/download?id=1oyknOulq_j0w9kzOKKIHdTLo5HphT99h`)
  hits Google's "can't scan this file for viruses" interstitial for large
  files (586MB) when fetched with plain `curl`/`wget` (returns an HTML page,
  not the file). The `gdown` PyPI package handles that confirmation step
  automatically. Downloaded and unzipped into
  `model_wrappers/cddd/default_model/` (matches what `init.sh` already checks
  for — no `init.sh` change needed).
- **`simson`: same approach works, hit a rate limit partway through.** The
  tokenizer (`pubchem_part_tokenizer.json`) was already present in the repo.
  The model checkpoint's Google Drive folder (same folder ID `init.sh`
  already references, `1EZhgntGlPzP9bmUMymaMVYO-DhBWJrxR`) contains 30
  per-epoch checkpoints plus the actual file the wrapper needs,
  `pretrained_best_model.pth` (confirmed via `simson_repo/trainer.py`, which
  saves that exact filename for whichever epoch had the lowest val loss during
  the original training run — not deducible from an epoch number alone, so
  this specific file is required, not just any epoch checkpoint). `gdown`
  downloaded 29/30 sibling epoch files in an earlier exploratory pass before
  hitting Google's anonymous per-file download quota ("many users have viewed
  this file") on a retry aimed at just the one needed file. This is a
  time-limited external rate limit, not a real blocker — retry
  `gdown --folder https://drive.google.com/drive/folders/1EZhgntGlPzP9bmUMymaMVYO-DhBWJrxR`
  later (quota typically resets in hours) and keep only
  `pretrained_best_model.pth`, or authenticate `gdown` with a real Google
  account to bypass the anonymous quota entirely.
- **`chemformer`: confirmed genuinely not scriptable headlessly.** The Box
  share link (`az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq`) serves a
  client-rendered SPA — `curl`/`wget` only get the app shell, not the folder
  listing or file contents, which load via authenticated XHR calls in-browser.
  Box's REST API can resolve a shared link, but requires an OAuth app token
  even for "anyone with the link" shares — there's no equivalent of `gdown`
  for Box without one. Needs an actual manual browser download, as originally
  documented.
- **`selformer`/`selformer-lite`** (the huggingface wrapper's two GDrive-gated
  variants): not attempted this session, to avoid compounding the Google Drive
  rate limit right before retrying `simson`. Same `gdown` approach should work
  once the quota clears — link is in `model_wrappers/huggingface/init.sh`.

## Data locations

- `data/processed/<source>/<endpoint>.csv` — cleaned raw data (17 endpoints).
- `data/prepared/reg_openadmet_*.joblib` — `Dataset` objects, ready for `embed.py`.
- `data/embedded/<dataset>/<model>.joblib` — local embedding cache.
- Modal Volume `ml-embedding-benchmark`, paths `/prepared` and `/embedded` —
  mirrors the above. As of this writing, has `fingerprints` (from the Modal
  run) plus `CLAMP`/`coati`/`mol2vec` (uploaded from local, complete-only —
  `huggingmolecules`/`unimol`'s in-progress partial files were deliberately
  **not** uploaded, to avoid confusing a partial result for a complete one).
  Re-upload once those two finish:
  ```
  modal volume put ml-embedding-benchmark data/embedded /embedded --force
  ```

## Local run recipe (if resuming a wrapper or starting a new one)

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

cd model_wrappers/<wrapper>
export INSTALL_DEP=true
pyenv shell "$(cat .python-version)"
python -m venv venv
source venv/bin/activate

pip install -r ../../base_requirements.txt
# rdkit/pandas/numpy<2/setuptools<81 aren't always pulled in transitively by
# a wrapper's own deps despite being needed -- see scripts/modal_embed.py's
# _install_base_requirements() for the full rationale.
pip install rdkit pandas "numpy<2.0.0" "setuptools<81"

source ./init.sh
pip install "numpy<2.0.0"   # re-assert; some wrappers' own init.sh (e.g. unimol) can upgrade numpy back past 2.0

cd ../../
export PYTHONPATH="$PYTHONPATH:.:model_wrappers/<wrapper>"
python -u embed.py --multirun "+experiment=$HYDRA_EXPERIMENT" '+dataset=glob(reg_openadmet_*)' ++cache=true
```

For `huggingface` specifically, add `'+model=glob(huggingface*,exclude=[*selformer*])'`
to the `embed.py` call to skip the two variants needing manual weight downloads.

## Status: embedding computation is done

Every wrapper (all 15) has produced embeddings for all 17 OpenADMET datasets,
across every model variant each wrapper defines. `molbert`/`graphfp`/
`graphmvp`/`mol-r` (Python 3.7) and `cddd` (Python 3.6) all ended up working
via pyenv-built legacy interpreters, not excluded as originally scoped.

`data/embedded` is now fully synced between local disk and the Modal Volume —
969 files each side, zero gap in either direction. Local and Modal had been
*complementary*, not unified (`fingerprints`/`gem`/`unimol`/`selformer*` only
ever ran on Modal; everything Python-3.6/3.7/3.9-based only ever ran locally).
Reconciled by diffing per-`(dataset, model)` file listings on each side (787
local ids vs 408 volume ids beforehand) rather than a blind `--force` sync in
either direction:
- 561 files existed locally but not on the Volume (33 models × 17 datasets,
  exactly the legacy-Python wrappers) — staged via symlinks into a temp dir
  mirroring just the gap, uploaded in one `modal volume put` call.
- 182 files existed on the Volume but not locally (`fingerprints`'
  `AtomPair`/`ECFP`/`TT` ×2 each + `GEM` + `unimolv1`/`unimolv2` = 9 models ×
  17, plus a partial `SELFormer`/`SELFormer-Lite` gap from before that moved
  to Modal) — `modal volume get` takes one path per call with no glob, so
  these were fetched with 182 individual calls rather than a bulk directory
  pull that would've redundantly re-downloaded everything already present.

## Next steps (not yet decided/started)

- Fix the regression-scoring path (`src/eval/supervised/train.py`'s
  GridSearchCV scorer and `src/eval/supervised/eval_metrics.py::evaluate()`
  both currently hardcode ROC-AUC/classification logic regardless of `task`)
  — flagged early in this work, not yet started. This is the actual blocker
  for turning these embeddings into comparable benchmark results.
