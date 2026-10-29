"""STARLING diffusion wrapper and ensemble metric extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import starling


MAX_SEQUENCE_LENGTH = 380   # starling.configs.MAX_SEQUENCE_LENGTH


def validate_sequence_length(seq: str) -> None:
    """Raise ValueError if seq exceeds STARLING's MAX_SEQUENCE_LENGTH."""
    if len(seq) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"sequence length {len(seq)} exceeds MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH}"
        )


def generate_ensembles(
    sequences: dict[str, str],
    *,
    conformations: int = 400,
    ionic_strength: float = 150.0,
    device: str | None = "cuda",
    verbose: bool = False,
) -> dict[str, object]:
    """Generate STARLING ensembles for a name->sequence mapping.

    Returns {name: starling.structure.ensemble.Ensemble}.
    """
    for name, seq in sequences.items():
        validate_sequence_length(seq)

    ensembles = starling.generate(
        sequences,
        conformations=conformations,
        ionic_strength=ionic_strength,
        device=device,
        return_structures=True,
        return_data=True,
        verbose=verbose,
        show_progress_bar=verbose,
        show_per_step_progress_bar=False,
    )
    return ensembles


def save_ensembles(
    ensembles: dict[str, object],
    output_dir: str | Path,
) -> None:
    """Save each Ensemble via its .save() method under output_dir/<name>.npz-style output."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, ens in ensembles.items():
        ens.save(str(output_dir / name))


def compute_rg_from_coords(coords: np.ndarray) -> np.ndarray:
    """Radius of gyration per conformer.

    coords shape: (n_conf, n_res, 3). Returns (n_conf,) in the same units as coords.
    Rg = sqrt(mean over residues of |r - r_mean|^2).
    """
    centered = coords - coords.mean(axis=1, keepdims=True)
    sq = (centered ** 2).sum(axis=-1)  # (n_conf, n_res)
    return np.sqrt(sq.mean(axis=1))


def compute_end_to_end_from_coords(coords: np.ndarray) -> np.ndarray:
    """End-to-end distance per conformer. Shape (n_conf,)."""
    d = coords[:, -1] - coords[:, 0]
    return np.sqrt((d ** 2).sum(axis=-1))


def compute_asphericity_from_coords(coords: np.ndarray) -> np.ndarray:
    """Asphericity per conformer, using gyration-tensor eigenvalues.

    asphericity = lambda_max - 0.5 * (lambda_mid + lambda_min)  (normalized by trace)
    For a spherical distribution all eigenvalues are equal and asphericity = 0.
    Shape: (n_conf,).
    """
    centered = coords - coords.mean(axis=1, keepdims=True)   # (n_conf, n_res, 3)
    # Gyration tensor S_ij = mean_k(r_ki * r_kj) per conformer → (n_conf, 3, 3)
    S = np.einsum("cki,ckj->cij", centered, centered) / centered.shape[1]
    eigvals = np.linalg.eigvalsh(S)   # ascending
    lam1, lam2, lam3 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
    trace = lam1 + lam2 + lam3
    # Normalize to avoid scale sensitivity.
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (lam3 - 0.5 * (lam1 + lam2)) / np.where(trace > 0, trace, 1.0)
    return a.astype(np.float32)


def extract_metrics(ensemble) -> dict[str, np.ndarray]:
    """Pull per-ensemble metrics out of a STARLING Ensemble.

    Uses Ensemble's built-in methods where available; falls back to coords for asphericity.
    """
    rg = np.asarray(ensemble.radius_of_gyration(), dtype=np.float32).ravel()
    ree = np.asarray(ensemble.end_to_end_distance(), dtype=np.float32).ravel()
    mean_dmap = np.asarray(ensemble.distance_maps(return_mean=True), dtype=np.float32)
    cmap = np.asarray(ensemble.contact_map(contact_thresh=11, return_mean=True), dtype=np.float32)

    # Asphericity needs raw coords. ensemble.trajectory returns a soursop SSProtein
    # object whose .traj attribute is an MDTraj Trajectory with .xyz in nanometers,
    # shape (n_conf, n_res, 3). Asphericity is scale-normalized so no unit conversion needed.
    traj = ensemble.trajectory
    if hasattr(traj, "traj") and hasattr(traj.traj, "xyz"):
        coords = np.asarray(traj.traj.xyz, dtype=np.float32)
    elif hasattr(traj, "xyz"):
        coords = np.asarray(traj.xyz, dtype=np.float32)
    else:
        coords = np.asarray(traj, dtype=np.float32)
    asph = compute_asphericity_from_coords(coords)

    return {
        "rg": rg,
        "end_to_end": ree,
        "mean_distance_map": mean_dmap,
        "contact_map": cmap,
        "asphericity": asph,
    }
