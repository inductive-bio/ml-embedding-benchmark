"""Type aliases shared across the SAND Lightning modules."""

from typing import Any, Dict, Literal, Optional

_BATCH_TYPE = Dict[str, Any]
_PL_STAGE_TYPE = Optional[Literal["train", "val", "test"]]
