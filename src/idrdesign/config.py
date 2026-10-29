"""Configuration object + YAML I/O + environment fingerprint for reproducibility."""
from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


VALID_COMPOSITION_MODES = ("none", "exact", "properties")


@dataclass
class Config:
    # Required
    target_fasta: str
    target_name: str
    output_dir: str

    # Design parameters
    n_designs: int = 10
    target_similarity: float = 0.95
    composition_mode: str = "none"

    # Ensemble parameters
    n_conformers: int = 400
    ionic_strength: float = 150.0

    # Safety caps
    max_iterations: int = 5000
    max_wall_seconds: float = 1800.0

    # Reproducibility & runtime
    seed: int = 0
    device: str = "cuda"

    # Populated at save time (not required at construction)
    env: dict = field(default_factory=dict)
    target_seq_sha256: str = ""

    def __post_init__(self):
        if self.composition_mode not in VALID_COMPOSITION_MODES:
            raise ValueError(
                f"composition_mode must be one of {VALID_COMPOSITION_MODES}, "
                f"got {self.composition_mode!r}"
            )


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(asdict(cfg), fh, sort_keys=False)


def load_config(path: str | Path) -> Config:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return Config(**data)


def env_fingerprint() -> dict[str, str]:
    """Capture package versions for reproducibility.

    Returns a dict of {package_name: version_string}.
    """
    fp: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("torch", "starling", "goose", "sparrow", "numpy", "pandas"):
        try:
            mod = __import__(pkg)
            # Coerce to plain str: e.g., torch.__version__ is a TorchVersion
            # (str subclass) that yaml.SafeDumper refuses to represent.
            fp[pkg] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            fp[pkg] = "not-installed"
    try:
        import torch
        fp["cuda_runtime"] = str(torch.version.cuda)
    except Exception:
        fp["cuda_runtime"] = "not-available"
    return fp


def sequence_sha256(seq: str) -> str:
    return hashlib.sha256(seq.encode("ascii")).hexdigest()
