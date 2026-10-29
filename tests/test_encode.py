"""Tests for idrdesign.encode."""
import pytest
import torch

from idrdesign.encode import cosine_similarity, l2_normalize


def test_l2_normalize_unit_norm():
    v = torch.tensor([3.0, 4.0])
    out = l2_normalize(v)
    assert torch.allclose(out.norm(), torch.tensor(1.0))
    assert torch.allclose(out, torch.tensor([0.6, 0.8]))


def test_l2_normalize_already_unit_is_stable():
    v = torch.tensor([1.0, 0.0, 0.0])
    out = l2_normalize(v)
    assert torch.allclose(out, v)


def test_cosine_similarity_identical_vectors_equals_one():
    a = torch.tensor([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal_equals_zero():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_opposite_equals_neg_one():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)


@pytest.mark.gpu
def test_encode_sequence_returns_1d_float_tensor():
    from idrdesign.encode import encode_sequence
    seq = "GSGSGSGSGSGSGSGSGSGS"
    emb = encode_sequence(seq)
    assert emb.ndim == 1
    assert emb.dtype in (torch.float32, torch.float16)
    assert emb.shape[0] > 0


@pytest.mark.gpu
def test_encode_sequence_identical_input_gives_identical_output():
    from idrdesign.encode import encode_sequence
    seq = "GSGSGSGSGSGSGSGS"
    a = encode_sequence(seq)
    b = encode_sequence(seq)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.gpu
def test_encode_sequence_different_seqs_differ():
    from idrdesign.encode import encode_sequence
    a = encode_sequence("G" * 30)
    b = encode_sequence("A" * 30)
    assert not torch.allclose(a, b, atol=1e-3)


@pytest.mark.gpu
def test_cosine_similarity_to_target_property_raw_value_zero_when_identical():
    import sparrow
    from idrdesign.encode import CosineSimilarityToTarget, encode_sequence

    seq = "GSGSGSGSGSGSGSGSGSGS"
    target_emb = encode_sequence(seq)

    prop = CosineSimilarityToTarget(target_emb)
    raw = prop.calculate_raw_value(sparrow.Protein(seq))
    # 1 - cos_sim is near-zero for identical sequence; tolerance accommodates
    # TF32 matmul noise on Ampere/Ada GPUs (observed ~1.2e-4 on L40S).
    assert raw == pytest.approx(0.0, abs=5e-4)


@pytest.mark.gpu
def test_cosine_similarity_to_target_property_raw_value_positive_when_different():
    import sparrow
    from idrdesign.encode import CosineSimilarityToTarget, encode_sequence

    target_seq = "G" * 30
    other_seq = "K" * 30
    target_emb = encode_sequence(target_seq)

    prop = CosineSimilarityToTarget(target_emb)
    raw = prop.calculate_raw_value(sparrow.Protein(other_seq))
    assert raw > 0.0


def test_cosine_similarity_to_target_property_has_target_value_zero():
    """Sanity: target_value is 0 so GOOSE drives raw_value toward 0."""
    from idrdesign.encode import CosineSimilarityToTarget

    fake_emb = torch.tensor([1.0, 2.0, 3.0])
    prop = CosineSimilarityToTarget(fake_emb)
    assert prop.target_value == 0.0
