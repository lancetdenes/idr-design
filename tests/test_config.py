"""Tests for idrdesign.config."""
from pathlib import Path


def test_config_defaults():
    from idrdesign.config import Config
    c = Config(target_fasta="in.fa", target_name="WT", output_dir="out")
    assert c.n_designs == 10
    assert c.target_similarity == 0.95
    assert c.composition_mode == "none"
    assert c.n_conformers == 400
    assert c.max_iterations == 5000
    assert c.max_wall_seconds == 1800
    assert c.seed == 0
    assert c.ionic_strength == 150.0
    assert c.device == "cuda"


def test_config_round_trip_yaml(tmp_path: Path):
    from idrdesign.config import Config, save_config, load_config

    c = Config(
        target_fasta="in.fa",
        target_name="WT",
        output_dir="out",
        n_designs=3,
        target_similarity=0.85,
    )
    p = tmp_path / "cfg.yaml"
    save_config(c, p)
    c2 = load_config(p)
    assert c2 == c


def test_config_rejects_unknown_composition_mode():
    import pytest
    from idrdesign.config import Config

    with pytest.raises(ValueError, match="composition_mode"):
        Config(
            target_fasta="in.fa",
            target_name="WT",
            output_dir="out",
            composition_mode="invalid",
        )


def test_env_fingerprint_contains_expected_keys():
    from idrdesign.config import env_fingerprint
    fp = env_fingerprint()
    for key in ("python", "torch", "starling", "goose"):
        assert key in fp


def test_config_with_env_fingerprint_is_yaml_safe(tmp_path: Path):
    # torch.__version__ is a TorchVersion (str subclass) that SafeDumper rejects
    # unless env_fingerprint coerces values to plain str.
    from idrdesign.config import Config, env_fingerprint, load_config, save_config

    c = Config(target_fasta="in.fa", target_name="WT", output_dir="out")
    c.env = env_fingerprint()
    p = tmp_path / "cfg.yaml"
    save_config(c, p)
    c2 = load_config(p)
    assert c2.env == c.env
    for v in c2.env.values():
        assert type(v) is str
