"""Per-stage torchmetrics wrapper used by the SAND Lightning training framework."""

from typing import List, Tuple, Union

import lightning.pytorch as pl
import torch
import torchmetrics
from torch.nn import ModuleDict

from sand.lit import _BATCH_TYPE, _PL_STAGE_TYPE


class Metric(pl.LightningModule):
    """Per-stage wrapper around a torchmetrics Metric.

    Clones the provided metric once per active stage (train/val/test) and
    dispatches updates and logging calls to the correct clone depending on the
    current Lightning stage.
    """

    def __init__(
        self,
        metric: torchmetrics.Metric,
        name: str,
        log_on_step_stages: List[_PL_STAGE_TYPE] = ["train"],
        log_on_epoch_stages: List[_PL_STAGE_TYPE] = ["val", "test"],
        prog_bar: bool = False,
    ) -> None:
        """Initialise the per-stage metric wrapper.

        Args:
            metric (torchmetrics.Metric): Base metric to clone per active stage.
            name (str): Display name used when logging this metric.
            log_on_step_stages (List[_PL_STAGE_TYPE]): Stages where the metric is
                logged at each step. Defaults to ["train"].
            log_on_epoch_stages (List[_PL_STAGE_TYPE]): Stages where the metric is
                accumulated and logged at epoch end. Defaults to ["val", "test"].
            prog_bar (bool): Whether to show this metric in the progress bar.
                Defaults to False.
        """
        super().__init__()
        self.name = name
        self.stage_map = {"train": "TRAIN", "val": "VAL", "test": "TEST"}
        self.stages = [
            stage
            for stage in ["train", "val", "test"]
            if stage in (log_on_step_stages + log_on_epoch_stages)
        ]
        self.metric_stage_dict = ModuleDict(
            {self.stage_map[stage]: metric.clone() for stage in self.stages}
        )
        self.log_on_step_stages = log_on_step_stages
        self.log_on_epoch_stages = log_on_epoch_stages
        self.prog_bar = prog_bar

    def __getitem__(self, stage: _PL_STAGE_TYPE) -> torchmetrics.Metric:
        """Return the metric clone for the given stage.

        Args:
            stage (_PL_STAGE_TYPE): One of ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            torchmetrics.Metric: The metric instance assigned to that stage.
        """
        metric = self.metric_stage_dict[self.stage_map[stage]]
        return metric

    def __call__(
        self,
        stage: _PL_STAGE_TYPE,
        **input: Union[_BATCH_TYPE, Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Update the metric for the given stage.

        For step-logging stages the metric is called directly (computes and
        resets each step); for epoch-logging stages only ``update`` is called
        so values accumulate until epoch end.

        Args:
            stage (_PL_STAGE_TYPE): Current Lightning stage.
            **input: Keyword arguments forwarded to the underlying metric.
        """
        metric = self[stage]
        if stage in self.log_on_step_stages:
            metric(**input)
        else:
            metric.update(**input)
