"""
Base Lightning module
"""

import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import lightning.pytorch as pl
import torch
import yaml
from jsonargparse import ArgumentParser
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH

from sand.lit import _BATCH_TYPE, _PL_STAGE_TYPE
from sand.lit.output import Output

__all__ = ["Module"]


class Module(pl.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        outputs: Union[List[Output], Output],
        sync_loss_log: bool = False,
        lr: float = 0.0005,
        min_lr: float = 0.00001,
        weight_decay: float = 0.05,
        warmup_iters: int = 10000,
        decay_iters: int = 1_000_000,
    ) -> None:
        """Pytorch Lightning Module wrapper for steps, logging etc.

        Args:
            backbone (torch.nn.Module): Torch module that takes a batch dictionary as
                input and returns an updated dictionary with predictions.
            outputs (Union[Sequence[Output], Output], optional): Sequence of
                Outputs that defines the outputs/tasks with their losses, metrics, etc.
            sync_loss_log (bool, optional): Whether to synchronize total loss log over
                epochs over devices. Turn this on when using ddp.
            lr (float, optional): Peak learning rate. Defaults to 0.0005.
            min_lr (float, optional): Minimum learning rate after decay. Defaults to
                0.00001.
            weight_decay (float, optional): AdamW weight decay. Defaults to 0.05.
            warmup_iters (int, optional): Number of linear warmup steps. Defaults to
                10000.
            decay_iters (int, optional): Number of steps over which the learning rate
                cosine-decays to `min_lr`. Defaults to 1_000_000.
        """
        super().__init__()
        self.backbone = backbone
        if not isinstance(outputs, Sequence):
            outputs = [outputs]
        if sync_loss_log:
            for output in outputs:
                output.sync_metric = True
        outputs = torch.nn.ModuleList(outputs)
        self.outputs = outputs
        self.sync_loss_log = sync_loss_log
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create and return the AdamW optimizer for the backbone parameters.

        Returns:
            torch.optim.Optimizer: Configured AdamW optimizer.
        """
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def update_lr(self, optimizer: torch.optim.Optimizer) -> None:
        """Update the learning rate using a warmup-then-cosine-decay schedule.

        Applies linear warmup from 0 to ``lr`` over ``warmup_iters`` steps,
        followed by cosine decay from ``lr`` to ``min_lr`` over the remaining
        steps up to ``decay_iters``.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer whose parameter-group
                learning rates are updated in-place.
        """
        if self.trainer.global_step < self.warmup_iters:
            lr = (
                self.lr
                * (float(self.trainer.global_step + 1))
                / (self.warmup_iters + 1)
            )
        elif self.trainer.global_step < self.decay_iters:
            decay_ratio = (float(self.trainer.global_step) - self.warmup_iters) / (
                self.decay_iters - self.warmup_iters
            )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.lr - self.min_lr)
        else:
            lr = self.min_lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer, optimizer_closure: Callable[[], None]) -> None:
        """Perform an optimizer step and immediately update the learning rate.

        Args:
            epoch (int): Current epoch index (passed through by Lightning).
            batch_idx (int): Current batch index within the epoch.
            optimizer (torch.optim.Optimizer): Optimizer to step.
            optimizer_closure (Callable): Closure that computes the loss.
        """
        optimizer.step(closure=optimizer_closure)
        self.update_lr(optimizer)

    def process_outputs(
        self,
        batch: _BATCH_TYPE,
        stage: _PL_STAGE_TYPE,
        dataloader_idx: int = 0,
        total_loss_log_offset: float = 0.0,
        subset: Optional[Sequence[str]] = None,
        add_dataloader_idx_to_total_loss: bool = False,
    ) -> torch.Tensor:
        """Forward all registered outputs, accumulate and log the total loss.

        Args:
            batch (_BATCH_TYPE): Current batch.
            stage (_PL_STAGE_TYPE): Current Lightning stage.
            dataloader_idx (int): Index of the current dataloader. Defaults to 0.
            total_loss_log_offset (float): Value added to the accumulated loss
                before logging. Defaults to 0.0.
            subset (Optional[Sequence[str]]): If provided, only process outputs
                whose ``name`` is in this set. Defaults to None (all outputs).
            add_dataloader_idx_to_total_loss (bool): Whether to append the
                dataloader index to the total-loss log key. Defaults to False.

        Returns:
            torch.Tensor: Total weighted loss summed across all active outputs.
        """
        total_loss = total_loss_log_offset
        for output in self.outputs:
            if subset is not None and output.name not in subset:
                continue
            weighted_loss = output(
                batch, stage=stage, log_func=self.log, dataloader_idx=dataloader_idx
            )
            total_loss = total_loss + weighted_loss
        if total_loss != 0:
            self.log(
                name=f"{stage}_loss",
                value=total_loss,
                on_step=(stage == "train"),
                on_epoch=(stage != "train"),
                prog_bar=True,
                batch_size=batch.get("batch_size", None),
                add_dataloader_idx=add_dataloader_idx_to_total_loss,
                sync_dist=self.sync_loss_log and stage != "train",
            )
        return total_loss

    def forward(
        self, inputs: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> Union[Dict[str, Any], torch.Tensor]:
        """Run the backbone on the given inputs.

        Args:
            inputs (Dict[str, Any]): Input dictionary passed to the backbone.
            *args: Additional positional arguments forwarded to the backbone.
            **kwargs: Additional keyword arguments forwarded to the backbone.

        Returns:
            Union[Dict[str, Any], torch.Tensor]: Backbone output.
        """
        pred = self.backbone(inputs, *args, **kwargs)
        return pred

    def step(
        self,
        batch: _BATCH_TYPE,
        stage: _PL_STAGE_TYPE,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """Run a single forward pass and compute the loss for any stage.

        Args:
            batch (_BATCH_TYPE): Current batch.
            stage (_PL_STAGE_TYPE): Lightning stage (``"train"``, ``"val"``,
                or ``"test"``).
            batch_idx (int): Index of the batch within the current epoch.
            dataloader_idx (int): Index of the current dataloader. Defaults to 0.

        Returns:
            torch.Tensor: Total weighted loss.
        """
        batch = self(batch)
        loss = self.process_outputs(batch, stage=stage, dataloader_idx=dataloader_idx)
        return loss

    def training_step(
        self, batch: _BATCH_TYPE, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Execute one training step.

        Args:
            batch (_BATCH_TYPE): Current training batch.
            batch_idx (int): Batch index within the epoch.
            dataloader_idx (int): Dataloader index. Defaults to 0.

        Returns:
            torch.Tensor: Training loss.
        """
        loss = self.step(
            batch=batch,
            stage="train",
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        return loss

    def validation_step(
        self, batch: _BATCH_TYPE, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Execute one validation step.

        Args:
            batch (_BATCH_TYPE): Current validation batch.
            batch_idx (int): Batch index within the epoch.
            dataloader_idx (int): Dataloader index. Defaults to 0.

        Returns:
            torch.Tensor: Validation loss.
        """
        loss = self.step(
            batch=batch, stage="val", batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )
        return loss

    def test_step(
        self, batch: _BATCH_TYPE, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Execute one test step.

        Args:
            batch (_BATCH_TYPE): Current test batch.
            batch_idx (int): Batch index within the epoch.
            dataloader_idx (int): Dataloader index. Defaults to 0.

        Returns:
            torch.Tensor: Test loss.
        """
        loss = self.step(
            batch=batch,
            stage="test",
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        return loss

    def on_fit_end(self) -> None:
        super().on_fit_end()

    @classmethod
    def instantiate_model_from_config(
        cls, config_path: str, skip_check: bool = True
    ) -> pl.LightningModule:
        """Instantiate a model from a YAML config file without loading weights.

        Args:
            config_path (str): Path to a ``config.yaml`` file whose ``model``
                key contains the constructor arguments.
            skip_check (bool): Skip jsonargparse validation. Defaults to True.

        Returns:
            pl.LightningModule: Instantiated (untrained) model.
        """
        with open(config_path) as yaml_file:
            config = yaml.load(yaml_file, Loader=yaml.FullLoader)  # nosec
            model_config = config["model"]
            model_config = {"model": model_config}
            parser = ArgumentParser()
            parser.add_class_arguments(cls, "model")
            cfg = parser.parse_object(model_config, _skip_validation=skip_check)
            cfg = parser.instantiate_classes(cfg)
            model = cfg.model
        return model

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: _PATH,
        config_path: _PATH = None,
        map_location: _MAP_LOCATION_TYPE = None,
        skip_check: bool = True,
    ) -> pl.LightningModule:
        """Load model weights from a checkpoint, optionally using a separate config.

        If ``config_path`` is not provided, the method looks for
        ``config.yaml`` two directory levels above the checkpoint.

        Args:
            checkpoint_path (_PATH): Path to the ``.ckpt`` checkpoint file.
            config_path (_PATH, optional): Path to the config YAML. If None,
                inferred from the checkpoint path. Defaults to None.
            map_location (_MAP_LOCATION_TYPE, optional): Device mapping for
                ``torch.load``. Defaults to None.
            skip_check (bool): Skip jsonargparse validation. Defaults to True.

        Returns:
            pl.LightningModule: Model loaded with the checkpoint's state dict.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(checkpoint_path)), "config.yaml"
            )
        model = cls.instantiate_model_from_config(
            config_path=config_path, skip_check=skip_check
        )
        state_dict = torch.load(checkpoint_path, map_location, weights_only=True)
        model.load_state_dict(state_dict["state_dict"])
        return model
