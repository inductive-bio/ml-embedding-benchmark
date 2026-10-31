# Datasets

The benchmark uses 17 regression endpoints from three OpenADMET blind challenges. All three challenges have concluded and released labeled test sets, so the original train/test partition is reused as-is (no re-splitting). `scripts/prepare_openadmet_data.py` downloads and cleans every endpoint into `data/processed/<source>/<endpoint>.csv` with columns `molecule_id, smiles, value, split`.

For every endpoint, rows with a missing label are dropped and SMILES are canonicalized with RDKit (rows that fail to parse are dropped). All endpoints are modeled on a log10 scale; endpoints released in linear units are transformed as described below, following each challenge's own evaluation code.

## ExpansionRx (9 endpoints)

Source: Hugging Face dataset [`openadmet/openadmet-expansionrx-challenge-data`](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data) (`expansion_data_train.csv`, `expansion_data_test.csv`).

Each assay column becomes its own single-task endpoint. Transforms follow the challenge tutorial, `log10((x + 1) * multiplier)`:

| Column | Endpoint | Transform |
|---|---|---|
| LogD | `LogD` | none (already log scale) |
| KSOL | `KSOL` | log10, multiplier 1e-6 |
| HLM CLint | `Log_HLM_CLint` | log10 |
| MLM CLint | `Log_MLM_CLint` | log10 |
| Caco-2 Permeability Papp A>B | `Log_Caco_Papp_AB` | log10, multiplier 1e-6 |
| Caco-2 Permeability Efflux | `Log_Caco_ER` | log10 |
| MPPB | `Log_Mouse_PPB` | log10 |
| MBPB | `Log_Mouse_BPB` | log10 |
| MGMB | `Log_Mouse_MPB` | log10 |

## Polaris antiviral ADMET and potency (7 endpoints)

Source: Polaris Hub datasets `asap-discovery/antiviral-admet-2025-unblinded` (HLM, KSOL, LogD, MDR1-MDCKII, MLM) and `asap-discovery/antiviral-potency-2025-unblinded` (pIC50 for SARS-CoV-2 Mpro and MERS-CoV Mpro). Loading them requires the `polaris-lib` client.

- Every ADMET endpoint except LogD is transformed with `log10(x + 1)`, matching the challenge's `clip_and_log_transform`. Potency values are already pIC50.
- Datapoints flagged as erroneous by the organizers are removed using the exclusion lists in [asapdiscovery/asap-polaris-blind-challenge-examples](https://github.com/asapdiscovery/asap-polaris-blind-challenge-examples) (`evaluation/data/exclusions/{admet,potency}.json`). Clone that repository into `data_repos/` before running the preparation script. Exclusions are matched by canonical SMILES and restricted to test-set rows, because dropping CXSMILES enhanced-stereo groups during canonicalization can make a flagged compound collide with an unrelated training-set stereoisomer.
- Molecules are released as CXSMILES. Canonicalization drops the enhanced-stereo annotations; none of the wrapped models consume them.

## PXR (1 endpoint)

Source: Hugging Face dataset [`openadmet/pxr-challenge-train-test`](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test), Activity track only. The training set is `pxr-challenge_TRAIN.csv`, and the test set is the union of the two unblinded phase files (`pxr-challenge_TEST_PHASE_1_UNBLINDED.csv` and `pxr-challenge_TEST_PHASE_2_UNBLINDED.csv`, 513 compounds). The endpoint is `pEC50`, which is already on a log scale.
