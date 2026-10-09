import pandas as pd

from omegaconf import DictConfig
from os.path import join
from hydra.utils import get_original_cwd
from typing import Tuple, List, Dict
from .types import Dataset
from rdkit import Chem

# Tuple of train, val, test indices
Splits = Dict[str, List[int]]


def local_csv_solver(path: str, target_column: str, root: str) -> Tuple[pd.DataFrame, Splits]:
    """Load one of the pre-cleaned OpenADMET endpoint CSVs produced by
    scripts/prepare_openadmet_data.py (columns: molecule_id, smiles, value, split)."""
    df = pd.read_csv(join(root, path))
    df = df.rename(columns={"value": target_column})

    splits = {
        "train": df[df["split"] == "train"].index.tolist(),
        "valid": [],
        "test": df[df["split"] == "test"].index.tolist(),
    }
    return df, splits


def build_dataset(name: str, task: str, raw_data: pd.DataFrame, splits: Splits) -> Dataset:
    raw_data['smiles'] = raw_data['smiles'].map(Chem.CanonSmiles)
    return Dataset(
        name=name,
        data=raw_data,
        splits=splits,
        task=task
    )


def load(dataset_config: DictConfig, raw_dir: str, use_hydra: bool = True) -> Dataset:
    if use_hydra:
        raw_dir = join(get_original_cwd(), raw_dir)

    if dataset_config.source.name == 'LOCAL_CSV':
        raw_data, splits = local_csv_solver(
            path=dataset_config.source.path,
            target_column=dataset_config.source.target_column,
            root=get_original_cwd() if use_hydra else '.',
        )
    else:
        raise ValueError(f"Unknown dataset source: {dataset_config.source.name}")

    return build_dataset(
        name=dataset_config.name,
        task=dataset_config.task,
        raw_data=raw_data,
        splits=splits
    )