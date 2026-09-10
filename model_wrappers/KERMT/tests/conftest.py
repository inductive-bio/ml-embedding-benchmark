# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MIT License

# Copyright (c) 2021 Tencent AI Lab.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import pytest
from pathlib import Path
from argparse import Namespace

@pytest.fixture(scope="session")
def data_dir():
    """
    Provides the absolute path to the test data directory.
    """
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def smis_only_csv_path(data_dir):
    """
    Provides the absolute path to the test smis_only.csv file.
    """
    return data_dir / "smis_only.csv"

@pytest.fixture(scope="session")
def smis_csv_path(data_dir):
    """
    Provides the absolute path to the test smis.csv file.
    """
    return data_dir / "smis.csv" 

@pytest.fixture(scope="session")
def finetune_data_dir(data_dir):
    return data_dir / "finetune"

@pytest.fixture(scope="session")
def finetune_args(finetune_data_dir):
    """
    Returns the generic arguments for the finetune task.
    """
    args = Namespace(parser_name='finetune', no_cache=True, gpu=0, batch_size=32, tensorboard=False, data_path=str(finetune_data_dir / "train.csv"), use_compound_names=False, max_data_size=None, features_only=False, features_generator=None, rdkit2D_normalization_type='fast', use_cuikmolmaker_featurization=False, features_path=[str(finetune_data_dir / "train.npz")], save_dir='model/frz_enc/fold_0', save_smiles_splits=False, checkpoint_dir=None,  dataset_type='regression', separate_val_path=str(finetune_data_dir / "val.csv"), separate_val_features_path=[str(finetune_data_dir / "val.npz")], separate_test_path=str(finetune_data_dir / "test.csv"), separate_test_features_path=[str(finetune_data_dir / "test.npz")], split_type='scaffold_balanced', split_sizes=[0.8, 0.1, 0.1], num_folds=1, folds_file=None, val_fold_index=None, test_fold_index=None, crossval_index_dir=None, crossval_index_file=None, seed=1, metric='mae', show_individual_scores=False, epochs=5, warmup_epochs=2.0, init_lr=0.0001, max_lr=0.0001, final_lr=2e-05, early_stop_epoch=1000, ensemble_size=1, dropout=0.0, activation='ReLU', ffn_hidden_size=700, ffn_num_layers=3, ffn_num_task_specific_layers=0, ffn_task_specific_hidden_size=None, weight_decay=0.0, select_by_loss=False, embedding_output_type='atom', self_attention=True, attn_hidden=4, attn_out=128, dist_coff=0.15, bond_drop_rate=0.1, distinct_init=False,  enbl_multi_gpu=False, n_trials=None, cuda=True, features_scaling=False, minimize_score=True, use_input_features=[str(finetune_data_dir / "train.npz")], num_lrs=1, fingerprint=False, wandb_project=None, wandb_run_name=None)
    return args