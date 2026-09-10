"""SANDModel: LightningModule subclass for training and inference with SAND."""

from typing import Any, Dict, Optional, Union, List
import torch
from sand.lit import _BATCH_TYPE
from sand.lit.module import Module
from sand.lit.output import Output
from sand.nn.backbone import IVFPQEncoderBackbone
from sand.mol_utils.pytorch import MolecularGraphDataloader, smiles_to_batch

# from torch_scatter import scatter_mul


class SANDModel(Module):
    """Lightning module for the SAND molecular embedding model.

    Extends :class:`~sand.lit.module.Module` with SAND-specific forward,
    training, validation, prediction, and encoding logic.
    """

    backbone: IVFPQEncoderBackbone

    def __init__(
        self,
        backbone: IVFPQEncoderBackbone,
        outputs: Union[List[Output], Output],
        sync_loss_log: bool = True,
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
            outputs (Union[Sequence[PLModelOutputs], PLModelOutputs], optional):
                Sequence of PLModelOutputs that defines the outputs/tasks with their
                losses, metrics, etc. . Defaults to ().
            lr (float, optional): Peak learning rate. Defaults to 0.0005.
            min_lr (float, optional): Minimum learning rate after decay. Defaults to
                0.00001.
            weight_decay (float, optional): AdamW weight decay. Defaults to 0.05.
            warmup_iters (int, optional): Number of linear warmup steps. Defaults to
                10000.
            decay_iters (int, optional): Number of steps over which the learning rate
                cosine-decays to `min_lr`. Defaults to 1_000_000.
        """
        super().__init__(
            backbone,
            outputs,
            sync_loss_log,
            lr,
            min_lr,
            weight_decay,
            warmup_iters,
            decay_iters,
        )

    def forward(self, batch: _BATCH_TYPE) -> _BATCH_TYPE:
        """Run the backbone on a batch.

        Args:
            batch: Batch dictionary passed to the backbone.

        Returns:
            Updated batch dictionary with backbone predictions added.
        """
        batch = self.backbone(batch)
        return batch

    def training_step(
        self, batch: _BATCH_TYPE, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Execute one training step, including optional codebook EMA update.

        Args:
            batch: Current training batch.
            batch_idx (int): Batch index within the epoch.
            dataloader_idx (int): Dataloader index. Defaults to 0.

        Returns:
            torch.Tensor: Training loss.
        """
        batch = self(batch)
        loss = self.process_outputs(batch, stage="train", dataloader_idx=dataloader_idx)
        # Codebook EMA is an explicit, train-only update. Run it here (before
        # backward) rather than in a post-backward hook: it uses detached tensor
        # values only, so it never touches the live autograd graph. Doing it in
        # a post-backward hook keeps prior-iteration AccumulateGrad nodes alive
        # under DDP (CUDA-stream-mismatch warning).
        if getattr(self.backbone, "quantize_emb", False):
            self.backbone.ema_update(batch)
        return loss

    def validation_step(
        self, batch: _BATCH_TYPE, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Execute one validation step.

        Args:
            batch: Current validation batch.
            batch_idx (int): Batch index within the epoch.
            dataloader_idx (int): Dataloader index. Defaults to 0.

        Returns:
            torch.Tensor: Validation loss.
        """
        batch = self(batch)
        loss = self.process_outputs(
            batch,
            stage="val",
            dataloader_idx=dataloader_idx,
            add_dataloader_idx_to_total_loss=True,
        )
        return loss

    def predict_step(
        self, batch: _BATCH_TYPE, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, Any]:
        """Run inference on a batch and return embeddings with SMILES.

        Args:
            batch: Batch dictionary; must contain molecule graph features.
                May contain a ``"smiles"`` key.
            batch_idx (int): Batch index. Defaults to 0.
            dataloader_idx (int): Dataloader index. Defaults to 0.

        Returns:
            dict: Dictionary with keys ``"emb"`` (embedding tensor) and
            ``"smiles"`` (list of SMILES strings or None).
        """
        emb = self.encode_batch(batch)
        smiles = batch.get("smiles", None)
        out = {"emb": emb, "smiles": smiles}
        return out

    def maybe_move_batch_to_gpu(self, batch):
        """Move a batch to the model's CUDA device if available.

        Args:
            batch: Batch object with a ``.to(device)`` method.

        Returns:
            Batch on the correct device.
        """
        if self.device.type == "cuda":
            batch = batch.to(self.device)
        return batch

    @torch.no_grad()
    def encode_batch(
        self, batch, emb_key: Optional[str] = None, batch_key: str = "batch"
    ) -> torch.Tensor:
        """Encode a pre-built graph batch into molecular embeddings.

        Runs the backbone encoder under ``torch.no_grad()``. If ``emb_key`` is
        ``"emb_quantized"``, the quantized IVF-PQ embedding is returned.

        Args:
            batch: Graph batch to encode. Will be moved to GPU if available.
            emb_key (str, optional): Key used to extract the embedding from the
                batch dict. Defaults to ``self.backbone.emb_key``.
            batch_key (str): Key identifying the batch assignment vector.
                Defaults to ``"batch"``.

        Returns:
            torch.Tensor: Embedding tensor of shape ``(N, emb_dim)``.
        """
        if emb_key is None:
            emb_key = self.backbone.emb_key
        batch = self.maybe_move_batch_to_gpu(batch)
        batch = self.backbone.encode(batch, batch_key=batch_key)
        if emb_key == "emb_quantized":
            batch = self.backbone.forward_batch_ivfpq(batch)
        emb = batch[emb_key]
        return emb

    @torch.no_grad()
    def encode_smiles(
        self,
        smiles: Union[List[str], str],
        emb_key: Optional[str] = None,
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> torch.Tensor:
        """Encode a list of SMILES strings into molecular embeddings.

        For small inputs the entire list is processed in a single batch; for
        larger inputs a :class:`MolecularGraphDataloader` is used to iterate
        in mini-batches.

        Args:
            smiles (Union[List[str], str]): One or more SMILES strings.
            emb_key (str, optional): Embedding key; defaults to
                ``self.backbone.emb_key``.
            batch_size (int): Number of molecules per mini-batch. Defaults to 256.
            num_workers (int): DataLoader worker processes. Defaults to 0.

        Returns:
            torch.Tensor: Embedding tensor of shape ``(N, emb_dim)``.
        """
        if isinstance(smiles, str):
            smiles = [smiles]
        if len(smiles) <= batch_size:
            batch = smiles_to_batch(smiles)
            batch = self.maybe_move_batch_to_gpu(batch)
            emb = self.encode_batch(batch, emb_key=emb_key)
            return emb
        else:
            dataloader = MolecularGraphDataloader(
                smiles_list=smiles,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            embeddings = []
            for batch in dataloader:
                batch = self.maybe_move_batch_to_gpu(batch)
                emb = self.encode_batch(batch, emb_key=emb_key)
                embeddings.append(emb)
            return torch.cat(embeddings, dim=0)
