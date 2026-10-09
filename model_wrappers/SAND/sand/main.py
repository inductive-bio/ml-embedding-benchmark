"""LightningCLI entry point for SAND model training."""

import pathlib
import subprocess
import logging
import warnings

import torch
import torch.autograd.graph
from lightning.pytorch.cli import ArgsType, LightningCLI
from sand.model import SANDModel
import torch._dynamo


class CLI(LightningCLI):
    """LightningCLI subclass that injects the current git commit hash as a CLI argument."""

    def add_arguments_to_parser(self, parser) -> None:
        """Add the current git commit hash as a ``--git_commit_hash`` argument.

        Args:
            parser: jsonargparse ArgumentParser instance provided by LightningCLI.
        """
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            cwd=pathlib.Path(__file__).parent,
        )
        git_hash = git_hash.stdout.decode().strip()
        parser.add_argument("--git_commit_hash", default=git_hash)

def filter_warnings() -> None:
    """Suppress known-benign warnings and noisy log messages before training."""
    torch._dynamo.config.suppress_errors = True
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined] 
    
    # DDP holds each parameter's AccumulateGrad node in its Reducer across
    # iterations by design, with a CUDA stream fixed at wrap time. On multi-GPU
    # runs this emits a benign per-iteration "AccumulateGrad node's stream does
    # not match" perf-hint warning (gradients are correct). The autograd-graph
    # docs sanction acknowledging this intentional multi-stream reuse; static_graph
    # is not usable here (incompatible with grad accumulation + the rotation-trick
    # dual gradient paths -> reducer assert). Guarded: only exists on newer torch.
    _set_warn = getattr(
        torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch", None
    )
    if _set_warn is not None:
        _set_warn(False)

    # Benign version-skew deprecation from Lightning's internal pytree usage
    # (uses the old `LeafSpec` API against a newer PyTorch). Not our code; filter
    # the noise until the Lightning dependency is bumped.
    warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\).*")

    # Silence Lightning's "install litlogger for cloud logging" promo tip. It's
    # emitted via `rank_zero_info` -> the stdlib logger below, at INFO level, not
    # a warning we can filter with `warnings`; raising the logger's level to
    # WARNING drops just this info-level chatter without hiding real warnings.
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)


def cli_main(args: ArgsType = None) -> None:
    """Configure global settings and launch the LightningCLI.

    Args:
        args (ArgsType, optional): Argument list override. Defaults to None
            (reads from ``sys.argv``).
    """
    filter_warnings()
    torch.set_float32_matmul_precision("high")

    CLI(
        model_class=SANDModel,
        args=args,
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    cli_main()
