"""Fine-tune a Uni-Mol checkpoint end-to-end on one OpenADMET endpoint and
score it on the same held-out test split the frozen-embedding path uses.

unimol_tools.MolTrain manages its own internal k-fold cross-validation and
early stopping (its `split`/`kfold`/`early_stopping` params), so this
concatenates finetune_data.load_finetune_splits()'s train+val rows into a
single fit() call rather than passing a separate validation set explicitly --
the val split is still requested from the shared loader (unused here) only so
every wrapper's adapter sees the same three-way split contract.

Must be run with this wrapper's own venv (model_wrappers/unimol/venv, Python
3.10, unimol_tools installed) -- see docs/openadmet_migration_status.md's
"Local run recipe" for how to activate it, or scripts/modal_embed_unimol.py
for the Modal image this reuses.

Usage:
    python model_wrappers/unimol/finetune.py --dataset reg_openadmet_expansionrx_LogD
    python model_wrappers/unimol/finetune.py --dataset reg_openadmet_pxr_pEC50 --model-name unimolv2
"""

import argparse
import glob
import logging as log
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.common.finetune_data import load_finetune_splits
from src.common.finetune_results import save_finetune_result
from src.common.utils import cuda_available

from unimol_tools import MolPredict, MolTrain

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run_finetune(
    dataset_name: str,
    dataset_config: dict,
    model_name: str,
    prepared_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    epochs: int,
    lr: float,
    batch_size: int,
    early_stopping: int,
    val_fraction: float,
    seed: int,
    run_tag: str = None,
) -> str:
    target_column = dataset_config["source"]["target_column"]

    train_df, val_df, test_df = load_finetune_splits(
        dataset_name=dataset_name,
        target_column=target_column,
        prepared_dir=prepared_dir,
        val_fraction=val_fraction,
        seed=seed,
    )
    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    # run_tag keeps a sweep run's checkpoint separate from the default
    # config's, rather than overwriting it.
    checkpoint_name = f"{model_name}_{run_tag}" if run_tag else model_name
    save_path = os.path.join(checkpoint_dir, dataset_name, checkpoint_name)
    os.makedirs(save_path, exist_ok=True)

    log.info(
        f"Fine-tuning '{model_name}' on '{dataset_name}': "
        f"train+val={len(fit_df)}, test={len(test_df)}"
    )

    trainer = MolTrain(
        task="regression",
        model_name=model_name,
        epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        early_stopping=early_stopping,
        smiles_col="smiles",
        target_cols=[target_column],
        save_path=save_path,
        use_cuda=cuda_available(),
    )
    trainer.fit(fit_df[["smiles", target_column]])

    predictor = MolPredict(load_model=save_path)
    # MolPredict.predict() returns only a bare y_pred array -- the DataFrame
    # with molecule_id/target/prediction columns is only ever written to
    # disk (via save_path), never returned. Passing molecule_id through as
    # an extra input column and reading that CSV back (rather than
    # indexing into test_df by position) keeps predictions correctly
    # aligned even if MolPredict's own SMILES validation were to drop a row.
    predict_dir = os.path.join(save_path, "test_predictions")
    predictor.predict(
        data=test_df[["molecule_id", "smiles", target_column]],
        save_path=predict_dir,
    )
    pred_csvs = sorted(glob.glob(os.path.join(predict_dir, "test.predict.*.csv")))
    if not pred_csvs:
        raise RuntimeError(f"MolPredict didn't write a prediction CSV under {predict_dir}")
    pred_df = pd.read_csv(pred_csvs[-1])

    out_path = save_finetune_result(
        dataset_name=dataset_name,
        model_name=model_name,
        molecule_id=pred_df["molecule_id"].to_numpy(),
        y_true=pred_df[target_column].to_numpy(),
        y_pred=pred_df[f"predict_{target_column}"].to_numpy(),
        metric_name=dataset_config["metric"],
        output_dir=output_dir,
        tag=run_tag,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset config name, e.g. 'reg_openadmet_expansionrx_LogD' (config/dataset/<name>.yaml)")
    parser.add_argument("--model-name", default="unimolv1", choices=["unimolv1", "unimolv2"])
    parser.add_argument("--config-dir", default="config/dataset")
    parser.add_argument("--prepared-dir", default="data/prepared")
    parser.add_argument("--checkpoint-dir", default="data/finetune_checkpoints")
    parser.add_argument("--output-dir", default="data/finetune_errors")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--early-stopping", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", default=None, help="Optional suffix (e.g. 'epochs30_lr5e-05') so a hyperparameter-sweep run is saved alongside the default config's result instead of overwriting it")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.config_dir, f"{args.dataset}.yaml")) as f:
        dataset_config = yaml.safe_load(f)

    stem = f"{args.model_name}_{args.run_tag}" if args.run_tag else args.model_name
    output_path = os.path.join(args.output_dir, args.dataset, f"{stem}__finetuned.joblib")
    if os.path.exists(output_path) and not args.override:
        log.info(f"Skipping (already exists): {output_path}")
        return

    out_path = run_finetune(
        dataset_name=args.dataset,
        dataset_config=dataset_config,
        model_name=args.model_name,
        prepared_dir=args.prepared_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        val_fraction=args.val_fraction,
        seed=args.seed,
        run_tag=args.run_tag,
    )
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
