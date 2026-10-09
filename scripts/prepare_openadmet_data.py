"""Download and clean the OpenADMET ExpansionRx, Polaris antiviral, and PXR datasets.

Writes two tiers into ./data/:
  data/raw/<source>/...        untouched pulls from HF / Polaris Hub
  data/processed/<source>/*.csv  one file per regression endpoint, columns:
                                  molecule_id, smiles, value, split

See docs/data_sources/*.md for the source-by-source rationale behind the choices
made here (which HF/Polaris repo to use, which transform, which exclusions).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"


def canon_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def save_endpoint(df, source, endpoint_name, id_col, smiles_col, value_col, split_col):
    sub = df[[id_col, smiles_col, value_col, split_col]].copy()
    sub.columns = ["molecule_id", "smiles", "value", "split"]
    sub["split"] = sub["split"].str.lower()

    n_total = len(sub)
    sub = sub.dropna(subset=["value"]).copy()
    n_labeled = len(sub)

    sub["smiles"] = sub["smiles"].map(canon_smiles)
    n_before_smi = len(sub)
    sub = sub.dropna(subset=["smiles"])
    n_after_smi = len(sub)
    if n_before_smi != n_after_smi:
        print(f"    ! dropped {n_before_smi - n_after_smi} rows with unparseable SMILES")

    out_dir = PROCESSED / source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{endpoint_name}.csv"
    sub.to_csv(out_path, index=False)

    n_train = int((sub["split"] == "train").sum())
    n_test = int((sub["split"] == "test").sum())
    print(
        f"    {endpoint_name}: {n_labeled}/{n_total} labeled -> {len(sub)} clean rows "
        f"(train={n_train}, test={n_test}) -> {out_path.relative_to(ROOT)}"
    )
    return sub


# ---------------------------------------------------------------------------
# ExpansionRx
# ---------------------------------------------------------------------------

# (log_scale, multiplier, short_name) — mirrors the official tutorial's conversion table
EXPANSIONRX_CONVERSION = {
    "LogD": (False, 1, "LogD"),
    "KSOL": (True, 1e-6, "KSOL"),
    "HLM CLint": (True, 1, "Log_HLM_CLint"),
    "MLM CLint": (True, 1, "Log_MLM_CLint"),
    "Caco-2 Permeability Papp A>B": (True, 1e-6, "Log_Caco_Papp_AB"),
    "Caco-2 Permeability Efflux": (True, 1, "Log_Caco_ER"),
    "MPPB": (True, 1, "Log_Mouse_PPB"),
    "MBPB": (True, 1, "Log_Mouse_BPB"),
    "MGMB": (True, 1, "Log_Mouse_MPB"),
}


def fetch_expansionrx():
    print("== ExpansionRx ==")
    train = pd.read_csv(
        "hf://datasets/openadmet/openadmet-expansionrx-challenge-data/expansion_data_train.csv"
    )
    test = pd.read_csv(
        "hf://datasets/openadmet/openadmet-expansionrx-challenge-data/expansion_data_test.csv"
    )

    raw_dir = RAW / "expansionrx"
    raw_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(raw_dir / "expansion_data_train.csv", index=False)
    test.to_csv(raw_dir / "expansion_data_test.csv", index=False)

    train = train.copy()
    test = test.copy()
    train["split"] = "train"
    test["split"] = "test"
    full = pd.concat([train, test], ignore_index=True)

    for col, (log_scale, multiplier, short_name) in EXPANSIONRX_CONVERSION.items():
        full[short_name] = full[col].astype(float)
        if log_scale:
            # log10((x + 1) * multiplier) -- matches expansion_tutorial.ipynb step 2
            full[short_name] = np.log10((full[short_name] + 1) * multiplier)
        save_endpoint(full, "expansionrx", short_name, "Molecule Name", "SMILES", short_name, "split")


# ---------------------------------------------------------------------------
# Polaris antiviral (ADMET + Potency)
# ---------------------------------------------------------------------------

EXCLUSIONS_DIR = ROOT / "data_repos/asap-polaris-blind-challenge-examples/evaluation/data/exclusions"


def load_exclusion_sets(json_name):
    """Return {target_label: {canonical_smiles, ...}} of known-bad datapoints."""
    with open(EXCLUSIONS_DIR / json_name) as fd:
        raw = json.load(fd)
    result = {}
    for target_label, entries in raw.items():
        smis = set()
        for entry in entries:
            c = canon_smiles(entry["CXSMILES"])
            if c:
                smis.add(c)
        result[target_label] = smis
    return result


def load_polaris_dataset(name):
    import polaris as po

    ds = po.load_dataset(name)
    zr = ds.zarr_root
    return pd.DataFrame({k: zr[k][:] for k in zr.keys()})


def fetch_polaris():
    print("== Polaris antiviral (ADMET + Potency) ==")

    admet_excl = load_exclusion_sets("admet.json")
    potency_excl = load_exclusion_sets("potency.json")

    raw_dir = RAW / "polaris"
    raw_dir.mkdir(parents=True, exist_ok=True)

    admet_df = load_polaris_dataset("asap-discovery/antiviral-admet-2025-unblinded")
    admet_df.to_csv(raw_dir / "antiviral_admet_2025_unblinded.csv", index=False)
    admet_df["_smiles_canon"] = admet_df["CXSMILES"].map(canon_smiles)

    for col in ["HLM", "KSOL", "LogD", "MDR1-MDCKII", "MLM"]:
        bad = admet_excl.get(col, set())
        df = admet_df.copy()
        if col != "LogD":
            # Match the official evaluation's clip_and_log_transform (every ADMET
            # target except LogD, which is already log-scale) -- see
            # docs/data_sources/polaris_antiviral.md §5.
            df[col] = np.log10(df[col] + 1)
        if bad:
            # Exclusions were curated against the *test*-set array of the original
            # blind competition. Canonicalizing away CXSMILES enhanced-stereo groups
            # can make a flagged compound's plain SMILES collide with an unrelated
            # train-set row (different Molecule Name/stereoisomer/batch/value) --
            # restrict to Set == "Test" so only the actually-flagged datapoints drop.
            before = df[col].notna().sum()
            mask = df["_smiles_canon"].isin(bad) & (df["Set"] == "Test")
            df.loc[mask, col] = np.nan
            after = df[col].notna().sum()
            print(f"    dropping {before - after} flagged bad datapoint(s) for {col}")
        save_endpoint(df, "polaris_admet", col.replace("/", "-"), "Molecule Name", "CXSMILES", col, "Set")

    potency_df = load_polaris_dataset("asap-discovery/antiviral-potency-2025-unblinded")
    potency_df.to_csv(raw_dir / "antiviral_potency_2025_unblinded.csv", index=False)
    potency_df["_smiles_canon"] = potency_df["CXSMILES"].map(canon_smiles)

    for col in ["pIC50 (SARS-CoV-2 Mpro)", "pIC50 (MERS-CoV Mpro)"]:
        bad = potency_excl.get(col, set())
        df = potency_df.copy()
        if bad:
            # See comment above the ADMET exclusion loop: restrict to Set == "Test".
            before = df[col].notna().sum()
            mask = df["_smiles_canon"].isin(bad) & (df["Set"] == "Test")
            df.loc[mask, col] = np.nan
            after = df[col].notna().sum()
            print(f"    dropping {before - after} flagged bad datapoint(s) for {col}")
        short_name = "pIC50_SARS_CoV_2_Mpro" if "SARS" in col else "pIC50_MERS_CoV_Mpro"
        save_endpoint(df, "polaris_potency", short_name, "Molecule Name", "CXSMILES", col, "Set")


# ---------------------------------------------------------------------------
# PXR
# ---------------------------------------------------------------------------


def fetch_pxr():
    print("== PXR (Activity track, pEC50) ==")
    base = "hf://datasets/openadmet/pxr-challenge-train-test"
    train = pd.read_csv(f"{base}/pxr-challenge_TRAIN.csv")
    p1 = pd.read_csv(f"{base}/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    p2 = pd.read_csv(f"{base}/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")

    raw_dir = RAW / "pxr"
    raw_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(raw_dir / "pxr-challenge_TRAIN.csv", index=False)
    p1.to_csv(raw_dir / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv", index=False)
    p2.to_csv(raw_dir / "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv", index=False)

    keep_cols = ["Molecule Name", "SMILES", "pEC50"]
    train = train[keep_cols].copy()
    train["split"] = "train"
    test = pd.concat([p1[keep_cols], p2[keep_cols]], ignore_index=True)
    test["split"] = "test"
    print(f"    test set: phase1={len(p1)} + phase2={len(p2)} = {len(test)} (expect 513)")

    full = pd.concat([train, test], ignore_index=True)
    save_endpoint(full, "pxr", "pEC50", "Molecule Name", "SMILES", "pEC50", "split")
    # Emax deliberately skipped by default -- see docs/data_sources/pxr.md §2


def main():
    fetch_expansionrx()
    fetch_polaris()
    fetch_pxr()
    print("\nDone.")


if __name__ == "__main__":
    main()
