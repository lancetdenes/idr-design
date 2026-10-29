"""IDR Ensemble-Based Sequence Design."""
from __future__ import annotations

import importlib

__version__ = "0.1.0"

__all__ = ["design_idr", "DesignResult", "gs_start", "__version__"]

# Resolved on first attribute access rather than at import time. This keeps the
# FINCHES design path (idrdesign.intermap_design) from pulling in torch and
# starling, which idrdesign.optimize imports at module level -- a deployment
# that only designs against intermaps then needs neither.
_LAZY = {
    "design_idr": "idrdesign.optimize",
    "DesignResult": "idrdesign.optimize",
    "gs_start": "idrdesign.optimize",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(__all__ + list(_LAZY))
