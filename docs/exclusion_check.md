I'll dig into both the paper and the repo. Let me start by pulling the source and the blog post.There's an `examples/openadmet.ipynb` in the repo. Let me look at the PCBA preprocessing first, since that's where any bioassay contamination would live.Now the key question — which benchmarks the decontamination list covers, and what's actually in the PCBA assay set. Let me check the checkpoint config and benchmark list.Now let me read the blog post and check what the ExpansionRx endpoints actually are.The blog confirms ExpansionRx. Now let me verify the PCBA assay list — that's where a PXR bioassay could plausibly hide, since PCBA is PubChem-derived.Zenodo and Valence storage are blocked from this sandbox, so I can't read the assay list directly. Let me research the PXR question from the other direction.I pulled the repo and traced the actual preprocessing code. Here's what I can confirm and what I can't.

**What's in the training set.** Monroe's 1,152 pretraining tasks are exactly: 62 graph-level semi-empirical properties and 3D descriptors from PM6/PubChemQC, 1,089 binary PubChem bioassay columns from Graphium's PCBA_1328, and one conformer-denoising objective. The bundled `checkpoint/config.json` shows `use_node_labels: False`, so even the PM6 atom-level targets were dropped. No experimental ADME endpoint is a training label anywhere in the model. That settles ExpansionRx cleanly, consistent with the blog post.

**Polaris.** Two separate things here, and only one of them is airtight.

Polaris *labels* were never used, since the only label sources are PM6 and PCBA. On molecules, `get_polaris_inchis()` in `monroe/preprocessing/pm6_prep.py` loads `benchmark.get_train_test_split()` and keeps only the **test** half, converts to InChI, and those InChIs get `ANTI JOIN`ed out of PM6 and routed into the validation set for PCBA. So Polaris test molecules are genuinely excluded. Polaris *training*-split molecules were not excluded and can appear in the pretraining corpus, carrying QC or PubChem bioassay labels rather than Polaris ones.

Two caveats on that. First, the exclusion list is exactly the 28 benchmarks in `monroe/assets/polaris_benchmarks.json` (six `polaris/adme-fang-*` plus 22 `tdcommons/*`). Any other Polaris benchmark, including the OpenADMET challenges hosted there, gets no filtering at all. Second, three of those 28 are `cyp2c9-veith`, `cyp2d6-veith`, and `cyp3a4-veith`, which derive from a large PubChem qHTS CYP panel. PCBA_1328 was built by scraping every PubChem assay with more than 6,000 active/inactive calls, which is precisely that assay's profile. If it's in there, the paper's claim that "there is no chance of label leakage, as the tasks differ" is too strong for those three tasks. It still wouldn't contaminate the test molecules, which were removed, but it's worth knowing.

**PXR.** This is the one I can't close from here. The OpenADMET PXR induction data is newly generated at Octant, 11,000+ compounds from Enamine Discovery Diversity 10 and FDA-approved sets plus analogs, run March–July 2026. Those measurements did not exist when PCBA (2023) or PM6 were assembled, so there is no PXR induction label in Monroe's training set.

The open question is whether a *different* PXR reporter assay is among the 1,089. PubChem carries NCATS/Tox21 qHTS PXR screens (AIDs 1346982 and 1346985 were used as auxiliary data by at least one challenge entrant), and those screens are large enough to clear Graphium's 6,000-compound cutoff. I couldn't verify: Zenodo and Valence storage are both blocked from my sandbox, and the assay list isn't committed to the repo, only regenerated into `pcba_stats.json` at preprocessing time. Separately, structural overlap with the PXR set is likely nonzero, since FDA-approved drugs are well covered by an 81M-molecule PubChem corpus, and none of it was filtered because OpenADMET isn't in the exclusion list.

To close it yourself, once you have the parquet:

```python
import pyarrow.parquet as pq
aids = [c.split("-")[1] for c in pq.read_schema("PCBA_1328_1564k.parquet").names
        if c.startswith("assayID-")]
```

then hit PubChem PUG REST for the assay descriptions and grep for `PXR`, `NR1I2`, or `pregnane`. If nothing matches, PXR is clean on both labels and tasks. For the structural side, hash the PXR challenge molecules to InChIKey and intersect against the pretraining shards, reporting counts only.

---

## Follow-up (2026-08-31): closed via direct download

Correction first: Zenodo and PubChem PUG REST are **not** blocked from this sandbox (unclear why the earlier attempt failed — worth retrying before assuming a resource is unreachable). Pulled `pcba_1328.zip` directly from `zenodo.org/records/8024997` (152.8MB confirmed complete — the doc above's "requires the parquet" is no longer a blocker) and ran the exact snippet suggested above.

**PXR is in PCBA_1328 — confirmed, not speculative.** `PCBA_1328_1564k.parquet` has 1,328 assay columns. `assayID-1346982` (PubChem AID 1346982, "Human pregnane X receptor (PXR) small molecule agonists, qHTS assay", confirmed via PUG REST: hPXR-mediated CYP3A4 induction, PXR-Luc HepG2 cells) **is present**, with 7,081 tested compounds (1,892 active / 5,189 inactive) — clears Monroe's own filter (`pcba_prep.py` defaults: min 100 pos, min 100 neg, min 1,000 valid) by more than an order of magnitude on every threshold, so it's essentially certain to survive into the final 1,089 kept assays (`pcba_stats.json` itself isn't regenerated here — no PM6 data locally — but nothing in this profile is close to a filter boundary). `assayID-1346985` (the second candidate AID) is **not** in the 1,328. So Monroe's task set does include a genuine PXR-agonism binary classification task, contradicting this doc's earlier "no PXR reporter assay confirmed" framing.

**Structural overlap, PXR — checked directly, not inferred.** Loaded `data/prepared/reg_openadmet_pxr_pEC50.joblib` (4,652 molecules: 4,139 train / 513 test) and intersected by InChIKey against the 7,081 AID-1346982-tested compounds. **223 molecules overlap (4.8% of the full PXR dataset) — all 223 fall in the train split, zero in the test split.** So held-out test-set metrics on `reg_openadmet_pxr_pEC50` are not contaminated by this: Monroe never saw pEC50 values (that label doesn't exist until 2026), and the compounds it did see PXR-relevant binary calls for are exactly the ones excluded from measuring test performance. The caveat is for **fine-tuning**, not the current frozen-embedding scores: if the encoder is later fine-tuned end-to-end on this endpoint's train split, those 223 molecules get a second, more direct exposure to PXR-relevant chemistry than the rest of train — worth a footnote if fine-tuned PXR results look unusually strong relative to other endpoints, so it doesn't get misread as generalization.

**CYP2C9/2D6/3A4-Veith caveat — also confirmed, not speculative.** The earlier doc flagged PubChem AID 1851 (the qHTS CYP panel TDC's `cyp2c9-veith`/`cyp2d6-veith`/`cyp3a4-veith` derive from) as plausibly in PCBA_1328 by profile match. Checked directly: `assayID-1851` **is present**, 16,544 tested compounds (6,700 active / 9,844 inactive), again clearing every filter threshold by a wide margin. Confirms the paper's blanket "no label leakage" claim doesn't hold for those three Polaris/TDC benchmarks specifically — though as before, this doesn't touch the OpenADMET endpoints this repo actually scores, since those three benchmarks aren't among them.

**Polaris (OpenADMET) — still open, but the blocker is now data volume, not access.** Structural overlap between `reg_openadmet_polaris_admet_*`/`reg_openadmet_polaris_potency_*` and Monroe's ~81M-molecule PM6 pretraining corpus is still unchecked. This is reachable now (confirmed: Zenodo record `8427533`, the PM6 descriptor source, responds fine), but it's 19 parquet files at 1.95–2.51GB each (~45.4GB total) plus a separate HuggingFace conformer-geometry dataset — genuinely too large to pull in this pass. A partial check (e.g. 1 of 19 shards, ~5% coverage) would only give a lower-bound overlap count, not a clean answer, so it wasn't attempted. If this matters enough to close, the tractable path is probably: get PM6's molecule-identifier column list from one shard's schema (cheap), then check whether InChIKey or CID is available and query in bulk rather than downloading everything.

**Bottom line on "is this an issue":** For the embeddings/results this session is actually producing (frozen-embedding extraction + held-out test scoring), no — the one concrete contamination found (PXR/AID-1346982) doesn't touch the test split, and Polaris structural overlap, while unverified, would only matter for fine-tuning (PM6 has no ADME/potency labels, so it can't leak labels into a held-out score, only give the encoder prior structural exposure). It becomes a real caveat worth stating explicitly the moment fine-tuning results are reported for `reg_openadmet_pxr_pEC50` (223 train molecules with prior PXR-relevant exposure) or any Polaris endpoint (unverified structural overlap, though PM6 carries no Polaris-relevant labels either way).

---

## Follow-up (2026-08-31, later): full-PCBA overlap (not just the PXR assay), and PM6 scoping

**Correction to the "zero overlap in PXR test" claim above.** That claim was scoped to the ~7,081 compounds tested in the *one* PXR-specific assay (AID-1346982). Re-ran the check against **all 1,328 PCBA_1328 assays** (i.e. "is this molecule a row in `PCBA_1328_1564k.parquet` at all," regardless of which assay(s) it was tested in) — the full parquet has one row per molecule (1,563,664 rows, `SMILES`/`CID`/`SID` + 1,328 `assayID-*` columns, 0 null/unparseable SMILES), canonicalized every row to InChIKey (RDKit, ~15 min single-threaded, 0 failures, 1,562,549 unique keys) and intersected against every `reg_openadmet_pxr_pEC50` and `reg_openadmet_polaris_{admet,potency}_*` molecule, split apart:

| dataset | split | N | overlap | % |
|---|---|---:|---:|---:|
| `reg_openadmet_pxr_pEC50` | train | 4,139 | 715 | 17.3% |
| `reg_openadmet_pxr_pEC50` | **test** | 513 | **12** | **2.3%** |
| `reg_openadmet_polaris_admet_HLM` | train | 301 | 4 | 1.3% |
| `reg_openadmet_polaris_admet_HLM` | test | 102 | 0 | 0.0% |
| `reg_openadmet_polaris_admet_KSOL` | train | 365 | 2 | 0.5% |
| `reg_openadmet_polaris_admet_KSOL` | test | 112 | 0 | 0.0% |
| `reg_openadmet_polaris_admet_LogD` | train | 352 | 3 | 0.9% |
| `reg_openadmet_polaris_admet_LogD` | test | 126 | 0 | 0.0% |
| `reg_openadmet_polaris_admet_MDR1_MDCKII` | train | 425 | 6 | 1.4% |
| `reg_openadmet_polaris_admet_MDR1_MDCKII` | test | 126 | 0 | 0.0% |
| `reg_openadmet_polaris_admet_MLM` | train | 303 | 5 | 1.7% |
| `reg_openadmet_polaris_admet_MLM` | test | 117 | 0 | 0.0% |
| `reg_openadmet_polaris_potency_pIC50_MERS_CoV_Mpro` | train | 901 | 1 | 0.1% |
| `reg_openadmet_polaris_potency_pIC50_MERS_CoV_Mpro` | test | 292 | 0 | 0.0% |
| `reg_openadmet_polaris_potency_pIC50_SARS_CoV_2_Mpro` | train | 842 | 0 | 0.0% |
| `reg_openadmet_polaris_potency_pIC50_SARS_CoV_2_Mpro` | test | 258 | 0 | 0.0% |

**So: PXR test is not actually clean against Monroe's full PCBA training set — 12 of 513 test molecules (2.3%) were tested in *some* PCBA assay** (not necessarily the PXR one; a follow-up attempt to identify exactly which assay(s) each of the 12 came from didn't finish in time and wasn't rerun — low priority, since the headline fact, that overlap exists in test at all, is what matters). This is genuine encoder-level structural/label exposure during pretraining for those 12 molecules — not an ADME or PXR-pEC50 label leak (PCBA labels are unrelated bioassay binary calls, and this pEC50 value didn't exist until 2026), but it does mean "held-out test" isn't perfectly held-out at the structure level for those 12 rows. **All 7 Polaris endpoints (5 ADMET + 2 Potency), by contrast, show zero test-split overlap against the full PCBA set** — the earlier doc's Polaris-vs-PCBA read holds up under the fuller check; train-split overlap exists (1-6 molecules per endpoint, consistent with PM6/PCBA being a large PubChem-derived corpus) but doesn't touch test.

**PM6 scoping — cheaper than previously thought, and the "blocked/45GB" framing above was wrong on both counts.**

- Zenodo *is* reachable (confirmed again) and *does* honor HTTP range requests (`curl --range` → `206 Partial Content`, `accept-ranges: bytes`) on the PM6 descriptor files at record `8427533` — 20 parquet shards (not 19; there's also a `pm6_random_splits.pt`), 43.9GB total (2.0-2.5GB each, smallest 996.5MB).
- Tested column-projected reads via `pyarrow.parquet` + `fsspec`'s HTTP filesystem directly against the smallest shard (`pm6_processed_20.parquet`, 996.5MB, 2,088,992 rows) — **no download needed for this, it's a live range-request read**:
  - Schema fetch (parquet footer only): 0.88s. Columns include `ordered_smiles`, **`inchi`** (full InChI string, not just SMILES — even better for exact matching than re-canonicalizing SMILES), and `pubchem_cid`.
  - `pubchem_cid` column alone, full shard: **6.66MB transferred, 4.2s** (0.67% of the 996.5MB file).
  - `inchi` column alone, full shard: **65.5MB transferred, 21.8s** (6.6% of the file — strings cost more than the fixed-width CID, but still small).
- Extrapolating the `inchi`-column ratio (6.6%) across all 20 shards by file size: **~2.9GB total transfer**, roughly 15-20 minutes done serially from this sandbox (faster if fetched concurrently across shards, which the range-request approach supports fine since each shard is an independent HTTP resource).

**Recommendation: this does not need Modal or a 45GB pull.** Extracting just the `inchi` (or `pubchem_cid`, if PubChem CID matching turns out sufficient — the Polaris/PXR datasets only carry SMILES, so InChI→InChIKey is the safe general path) column from all 20 shards via range requests, entirely from a local sandbox with an internet connection, is a ~2.9GB / ~15-20 min job — not attempted here per the directive (this fork was scoped to test the cheap path on one shard and report numbers, not execute the full 20-shard pull), but it's a small enough commitment that it seems reasonable to just run directly next time, no Modal volume upload or GPU/CPU fan-out required. If it's ever worth parallelizing (e.g. to cut the ~15-20 min further), the natural split is 20 independent per-shard column reads — trivially fits this repo's existing Modal fan-out pattern (`scripts/modal_embed_*.py`) if desired, but plain local concurrent HTTP requests (e.g. a `ThreadPoolExecutor` over the 20 shard URLs) would very likely finish in a couple of minutes without needing Modal at all, since the bottleneck here is 20 independent small transfers, not compute.

---

## Follow-up (2026-08-31, later still): PM6 overlap executed — real numbers, and PXR test is not clean

Ran the full 20-shard check the prior scoping fork only tested on 1 shard. Method: built a query set of 6,146 unique InChIKeys from every `reg_openadmet_pxr_pEC50` and `reg_openadmet_polaris_{admet,potency}_*` molecule (9,274 total rows across the 8 datasets, 0 canonicalization failures — a compound repeating across the 5 Polaris ADMET / 2 Potency endpoints, which share compound sets, collapses to one key), tagged with which (dataset, split, molecule_id) it came from. Then, for all 20 PM6 descriptor parquet shards at Zenodo record `8427533`, range-projected-read just the `inchi` + `pubchem_cid` columns (`fsspec` HTTP filesystem + `pyarrow.parquet.read_table(columns=[...])`, no full-file download), converted every InChI to InChIKey (`Chem.InchiToInchiKey`, ~230k/s, RDKit warnings suppressed), and checked membership against the query set — 8 shards fetched concurrently (`ThreadPoolExecutor(max_workers=8)`).

**Actual runtime: 670s (11.2 min) wall time**, close to the prior fork's 15-20 min estimate (download-bound: ~3MB/s effective per connection off Zenodo from this sandbox; concurrency across 8 shards helped but Zenodo/network throughput capped the gain). 83,746,835 total PM6 rows scanned (~84M, matching the paper's "~81M" to within shard-count rounding).

**Overlap by dataset / split** (molecule counts, not row counts — a molecule tested is a molecule tested):

| dataset | split | N | PM6 overlap | % |
|---|---|---:|---:|---:|
| `reg_openadmet_pxr_pEC50` | train | 4,139 | 888 | 21.5% |
| `reg_openadmet_pxr_pEC50` | **test** | 513 | **190** | **37.0%** |
| `reg_openadmet_polaris_admet_HLM` | train | 301 | 6 | 2.0% |
| `reg_openadmet_polaris_admet_HLM` | test | 102 | 0 | 0.0% |
| `reg_openadmet_polaris_admet_KSOL` | train | 365 | 8 | 2.2% |
| `reg_openadmet_polaris_admet_KSOL` | test | 112 | 0 | 0.0% |
| `reg_openadmet_polaris_admet_LogD` | train | 352 | 8 | 2.3% |
| `reg_openadmet_polaris_admet_LogD` | test | 126 | 0 | 0.0% |
| `reg_openadmet_polaris_admet_MDR1_MDCKII` | train | 425 | 11 | 2.6% |
| `reg_openadmet_polaris_admet_MDR1_MDCKII` | test | 126 | 0 | 0.0% |
| `reg_openadmet_polaris_admet_MLM` | train | 303 | 6 | 2.0% |
| `reg_openadmet_polaris_admet_MLM` | test | 117 | 0 | 0.0% |
| `reg_openadmet_polaris_potency_pIC50_MERS_CoV_Mpro` | train | 901 | 5 | 0.6% |
| `reg_openadmet_polaris_potency_pIC50_MERS_CoV_Mpro` | test | 292 | 0 | 0.0% |
| `reg_openadmet_polaris_potency_pIC50_SARS_CoV_2_Mpro` | train | 842 | 3 | 0.3% |
| `reg_openadmet_polaris_potency_pIC50_SARS_CoV_2_Mpro` | test | 258 | 0 | 0.0% |

**This materially changes the PXR read, and confirms the Polaris read.**

**Polaris: clean, as expected.** Zero test-split overlap against PM6 across all 7 endpoints (matching the earlier zero-overlap-against-PCBA finding) — train-side overlap is small (3-11 molecules, 0.3-2.6%). This is exactly what `pm6_prep.py::get_polaris_inchis()`'s anti-join is documented to do for the 28 benchmarks in `monroe/assets/polaris_benchmarks.json` — though the OpenADMET Polaris endpoints scored here aren't among those 28, so this clean result isn't from deliberate filtering, it's coincidental (PM6/PCBA are PubChem-derived corpora and this antiviral compound series apparently isn't PubChem-heavy).

**PXR: not clean — 37% of the held-out test split has direct structural exposure from PM6 pretraining.** This is an order of magnitude larger than the PCBA-alone finding (12/513, 2.3%) from the previous follow-up. Unlike Polaris, PXR was never covered by Monroe's exclusion list (only the 28 named benchmarks are anti-joined out of PM6/PCBA) — and unlike Polaris's antiviral series, the PXR dataset is explicitly built from "Enamine Discovery Diversity 10 and FDA-approved sets plus analogs" (per the earlier follow-up), i.e. deliberately drug-like/well-known chemical space that heavily overlaps an 81M-molecule PubChem-derived corpus. This is **not a pEC50 label leak** — PM6's targets are semi-empirical QC properties (`use_node_labels: False`, no experimental ADME data anywhere in PM6), so the model never saw this assay's actual values. But it is a genuine break in the train/test independence assumption at the structural level: over a third of the PXR "held-out" test set was part of the encoder's pretraining corpus, which the encoder could plausibly have memorized fine-grained structural features of, independent of any label. Combined with the earlier PCBA finding, PXR test molecules overlap Monroe's pretraining data via PM6 (37.0%) far more than via PCBA (2.3%) — the two aren't mutually exclusive counts (some molecules likely appear in both), but the totals aren't summed here since that would need per-molecule dedup across the two checks, not done.

**Bottom line, updated**: for **Polaris**, nothing changes — test-clean holds under both PCBA and PM6 checks, and this remains a low-priority caveat overall. For **PXR**, the "is this an issue" answer from the earlier follow-up needs revising: frozen-embedding test-set scores on `reg_openadmet_pxr_pEC50` (the ones already computed for Monroe in `data/errors/`, and for every other embedding in this benchmark scored against this same endpoint) should be read with this in mind — not as invalid (no label leak occurred), but as a benchmark whose "held-out" test split is, for over a third of its molecules, structurally not novel to Monroe's encoder. This is specific to Monroe (or any future model trained on PM6/PCBA) — it doesn't affect the other ~56 embeddings already scored against PXR in this repo, which have no such pretraining relationship to PubChem's PM6/PCBA corpora. Worth a footnote wherever `reg_openadmet_pxr_pEC50` results for Monroe specifically are reported or compared against other embeddings.