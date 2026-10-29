"""Shared pytest fixtures and markers."""
import pytest


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.gpu tests when no GPU is present."""
    if _gpu_available():
        return
    skip_gpu = pytest.mark.skip(reason="requires GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
