try:
    from ._huggingface import HuggingFaceCompatibleModule  # optional
except Exception:  # pragma: no cover
    HuggingFaceCompatibleModule = None

from ._darray import Darray
from ._filter import apply_transforms, iter_module
from ._training import (
    compute_metrics,
    init_module,
    make_train_step,
    maybe_checkpoint,
    maybe_do,
    maybe_write,
    metrics_to_host,
    Optimizer,
    setup_module_opts,
)


__all__ = [
    "HuggingFaceCompatibleModule",
    "Darray",
    "iter_module", 
    "apply_tranforms",
    "Optimizer",
    "setup_module_opts",
    "make_train_step",
    "compute_metrics",
    "metrics_to_host",
    "maybe_write",
    "maybe_checkpoint",
    "maybe_do",
    "init_module",
]
