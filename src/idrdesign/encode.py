"""STARLING encoder wrapper + cosine similarity utilities.

The paper's objective: maximize cosine similarity between the L2-normalized
STARLING sequence-encoder embeddings of candidate and target.
"""
from __future__ import annotations

import torch

import starling


DEFAULT_IONIC_STRENGTH = 150  # mM; matches starling.configs.DEFAULT_IONIC_STRENGTH


def ensure_eval_mode(device: str | torch.device | None = None) -> None:
    """Put STARLING's models in eval mode. Cheap, idempotent, call freely.

    STARLING ships them in training mode: ModelManager.load_models never calls
    .eval(), and torch.inference_mode() only disables autograd -- it does not
    change module mode. 511 of 712 submodules stay training, which leaves this
    branch of SequenceEncoder.forward live:

        if self.training:
            mask_ionic = torch.rand(...) < 0.2
            ionic_strengths[mask_ionic] = 0

    i.e. roughly one call in five silently discards the ionic strength it was
    given. Measured displacement is about 37% of the full 0-500 mM range, and
    it makes the encoder nondeterministic on a fixed sequence.

    The device is resolved exactly as STARLING resolves it, so this never pins
    the models to a device the caller did not ask for.
    """
    from starling import utilities
    from starling.inference.generation import model_manager

    vae, diffusion = model_manager.get_models(device=utilities.check_device(device))
    if vae.training or diffusion.training:
        vae.eval()
        diffusion.eval()


def l2_normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return v / ||v||_2 along the last axis."""
    return v / (v.norm(p=2, dim=-1, keepdim=True) + eps)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors, returned as a Python float."""
    na = l2_normalize(a)
    nb = l2_normalize(b)
    return float((na * nb).sum().item())


def encode_sequence(
    seq: str,
    ionic_strength: float = DEFAULT_IONIC_STRENGTH,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return the aggregated (mean-pooled) STARLING encoder embedding for `seq`.

    Shape: (D,) where D is the STARLING latent dim.
    Kept on CPU by default (we only move to GPU for batched inner loops later).
    """
    ensure_eval_mode(device)
    out = starling.sequence_encoder(
        {"q": seq},
        ionic_strength=ionic_strength,
        aggregate=True,
        device=device,
        return_on_cpu=True,
    )
    return out["q"]


def encode_sequence_perresidue(
    seq: str,
    ionic_strength: float = DEFAULT_IONIC_STRENGTH,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return the per-residue (un-pooled) STARLING encoder embedding for `seq`.

    Shape: (L, D). Same forward pass as `encode_sequence`; only the final
    mean-pool over residues is skipped (overhead ~nil). This is the (L, D)
    conditioning STARLING actually uses to generate the ensemble, so matching it
    per-residue is a strictly richer objective than matching the mean-pool.
    """
    ensure_eval_mode(device)
    out = starling.sequence_encoder(
        {"q": seq},
        ionic_strength=ionic_strength,
        aggregate=False,
        device=device,
        return_on_cpu=True,
    )
    return out["q"]


def per_residue_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Mean over residues of the per-position cosine similarity between two
    (L, D) embeddings. Requires equal length / positional correspondence."""
    an = a / (a.norm(p=2, dim=-1, keepdim=True) + eps)
    bn = b / (b.norm(p=2, dim=-1, keepdim=True) + eps)
    return float((an * bn).sum(dim=-1).mean().item())


from goose.backend.optimizer_properties import ProteinProperty  # noqa: E402


class CosineSimilarityToTarget(ProteinProperty):
    """GOOSE ProteinProperty: raw_value = 1 - cos_sim(encode(seq), target_emb).

    GOOSE minimizes (raw_value - target_value). We set target_value=0 so the
    optimizer drives the candidate embedding toward the target embedding in
    cosine-similarity space.
    """

    can_be_linear_profile = False
    calculate_in_batch = False

    def __init__(
        self,
        target_embedding: torch.Tensor,
        *,
        weight: float = 1.0,
        ionic_strength: float = DEFAULT_IONIC_STRENGTH,
    ):
        super().__init__(target_value=0.0, weight=weight)
        self._target_emb_normed = l2_normalize(target_embedding.detach().clone())
        self._ionic_strength = ionic_strength

    def calculate_raw_value(self, protein) -> float:  # protein: sparrow.Protein
        seq = protein.sequence
        emb = encode_sequence(seq, ionic_strength=self._ionic_strength)
        emb_normed = l2_normalize(emb)
        cos_sim = float((self._target_emb_normed * emb_normed).sum().item())
        return 1.0 - cos_sim


class PerResidueCosineSimilarityToTarget(ProteinProperty):
    """GOOSE ProteinProperty: raw_value = 1 - per_residue_cosine(encode(seq), target).

    Parallels CosineSimilarityToTarget but matches the target's per-residue
    (L, D) embedding position-by-position instead of the mean-pooled (D,) vector.
    GOOSE minimizes raw_value -> 0, driving the candidate toward the target in
    per-residue cosine space. Requires candidate length == target length.
    """

    can_be_linear_profile = False
    calculate_in_batch = False

    def __init__(
        self,
        target_perres_embedding: torch.Tensor,
        *,
        weight: float = 1.0,
        ionic_strength: float = DEFAULT_IONIC_STRENGTH,
    ):
        super().__init__(target_value=0.0, weight=weight)
        self._target = target_perres_embedding.detach().clone()  # (L, D)
        self._ionic_strength = ionic_strength

    def calculate_raw_value(self, protein) -> float:  # protein: sparrow.Protein
        emb = encode_sequence_perresidue(protein.sequence, ionic_strength=self._ionic_strength)
        if emb.shape[0] != self._target.shape[0]:
            return 1.0  # length mismatch -> per-residue cosine undefined; worst score
        return 1.0 - per_residue_cosine(emb, self._target)
