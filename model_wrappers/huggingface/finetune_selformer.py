"""Fine-tune SELFormer (or SELFormer-Lite) end-to-end on one OpenADMET
endpoint and score it on the same held-out test split the frozen-embedding
path uses.

Kept separate from a generic huggingface fine-tune file since SELFormer is
the only huggingface-bundled model (of MolFormer/ChemBERTa/ChemGPT/ChemFM/
SELFormer) with a documented fine-tuning path -- mirrors how
scripts/modal_embed_selformer.py is already split out from the general
scripts/modal_embed.py for the same reason.

Reimplements selformer_repo/train_regression_model.py's model/dataset classes
directly (rather than shelling out to that script) because that script: (1)
does its own train_val_test_split from a CSV path, incompatible with the
train/val/test DataFrames every wrapper here must share (finetune_data.py);
(2) never saves per-molecule test predictions, only aggregate MSE/RMSE/MAE;
(3) its smiles_to_selfies() (in prepare_finetuning_data.py) imports chemprop
+ pandarallel at module level just for an unused scaffold-split code path --
avoided here by calling to_selfies.to_selfies() directly per-row, which only
needs the `selfies` package (already in modal_embed_selformer.py's image).

Hyperparameter defaults below (epochs=50, lr=1e-5, batch_size=8, wd=0.1)
mirror train_regression_model.py's own argparse defaults exactly, same as
model_wrappers/unimol/finetune.py mirrors unimol_tools.MolTrain's defaults.

Usage:
    python model_wrappers/huggingface/finetune_selformer.py --dataset reg_openadmet_pxr_pEC50
    python model_wrappers/huggingface/finetune_selformer.py --dataset reg_openadmet_pxr_pEC50 --model-name SELFormer-Lite
"""

import argparse
import logging as log
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import MSELoss
from torch.utils.data import Dataset
from transformers import (
    BertPreTrainedModel,
    RobertaConfig,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "selformer_repo"))

from src.common.finetune_data import load_finetune_splits
from src.common.finetune_results import save_finetune_result
import to_selfies  # selformer_repo/to_selfies.py

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

MAX_LEN = 128
CHECKPOINT_DIRS = {
    "SELFormer": os.path.join(REPO_ROOT, "model_wrappers/huggingface/model_weights/SELFormer"),
    "SELFormer-Lite": os.path.join(REPO_ROOT, "model_wrappers/huggingface/model_weights/SELFormer-Lite"),
}


class SELFIESTransformers_For_Regression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs[0])
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss = MSELoss()(logits.squeeze(), labels.squeeze())
            outputs = (loss,) + outputs
        return outputs


class SELFIESTransfomers_Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.examples = tokenizer(
            text=texts, text_pair=None, truncation=True,
            padding="max_length", max_length=MAX_LEN, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        item = {key: self.examples[key][index] for key in self.examples}
        item["label"] = self.labels[index]
        return item


def smiles_df_to_selfies(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row port of selformer_repo/prepare_finetuning_data.py::smiles_to_selfies,
    minus its chemprop/pandarallel dependency (unused here -- we already have
    our own train/val/test split, not its scaffold-split path)."""
    df = df.copy()
    df["selfies"] = df["smiles"].map(to_selfies.to_selfies)
    # to_selfies() returns the input unchanged on an EncoderError -- the same
    # signal prepare_finetuning_data.py uses to drop unencodable rows.
    failed = df["selfies"] == df["smiles"]
    if failed.any():
        log.warning(f"Dropping {failed.sum()} row(s) that failed SMILES->SELFIES encoding")
    return df[~failed].reset_index(drop=True)


def run_finetune(
    dataset_name: str,
    dataset_config: dict,
    model_name: str,
    prepared_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    epochs: int,
    lr: float,
    train_batch_size: int,
    validation_batch_size: int,
    wd: float,
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
    train_df = smiles_df_to_selfies(train_df)
    val_df = smiles_df_to_selfies(val_df)
    test_df = smiles_df_to_selfies(test_df)

    checkpoint_name = f"{model_name}_{run_tag}" if run_tag else model_name
    save_path = os.path.join(checkpoint_dir, dataset_name, checkpoint_name)
    os.makedirs(save_path, exist_ok=True)

    log.info(
        f"Fine-tuning '{model_name}' on '{dataset_name}': "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    pretrained_dir = CHECKPOINT_DIRS[model_name]
    config = RobertaConfig.from_pretrained(pretrained_dir, num_labels=1)
    model = SELFIESTransformers_For_Regression.from_pretrained(pretrained_dir, config=config)
    tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_dir, do_lower_case=False)

    train_dataset = SELFIESTransfomers_Dataset(
        train_df["selfies"].astype(str).tolist(), train_df[target_column].astype(float).tolist(), tokenizer
    )
    val_dataset = SELFIESTransfomers_Dataset(
        val_df["selfies"].astype(str).tolist(), val_df[target_column].astype(float).tolist(), tokenizer
    )
    test_dataset = SELFIESTransfomers_Dataset(
        test_df["selfies"].astype(str).tolist(), test_df[target_column].astype(float).tolist(), tokenizer
    )

    training_args = TrainingArguments(
        output_dir=save_path,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=wd,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=validation_batch_size,
        disable_tqdm=True,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
    )
    trainer.train()
    trainer.save_model(save_path)

    raw_pred, _, _ = trainer.predict(test_dataset)
    y_pred = np.asarray([row[0] for row in raw_pred], dtype=float)

    out_path = save_finetune_result(
        dataset_name=dataset_name,
        model_name=model_name,
        molecule_id=test_df["molecule_id"].to_numpy(),
        y_true=test_df[target_column].to_numpy(dtype=float),
        y_pred=y_pred,
        metric_name=dataset_config["metric"],
        output_dir=output_dir,
        tag=run_tag,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset config name, e.g. 'reg_openadmet_pxr_pEC50' (config/dataset/<name>.yaml)")
    parser.add_argument("--model-name", default="SELFormer", choices=["SELFormer", "SELFormer-Lite"])
    parser.add_argument("--config-dir", default="config/dataset")
    parser.add_argument("--prepared-dir", default="data/prepared")
    parser.add_argument("--checkpoint-dir", default="data/finetune_checkpoints")
    parser.add_argument("--output-dir", default="data/finetune_errors")
    parser.add_argument("--epochs", type=int, default=50, help="Default: train_regression_model.py's own --num_epochs default")
    parser.add_argument("--lr", type=float, default=1e-5, help="Default: train_regression_model.py's own --lr default")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", default=None, help="Optional suffix so a hyperparameter-sweep run is saved alongside the default config's result instead of overwriting it")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    import yaml
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
        train_batch_size=args.train_batch_size,
        validation_batch_size=args.validation_batch_size,
        wd=args.wd,
        val_fraction=args.val_fraction,
        seed=args.seed,
        run_tag=args.run_tag,
    )
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
