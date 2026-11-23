import pytest

from src.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["target"]["horizon_months"] == 6
    assert cfg["evaluation"]["primary_metric"] == "pr_auc"


def test_split_must_be_in_time_order(tmp_path):
    bad = tmp_path / "c.yaml"
    text = open("config.yaml").read().replace('train_cutoff: "2025-05-01"', 'train_cutoff: "2027-01-01"')
    bad.write_text(text)
    with pytest.raises(ValueError):
        load_config(bad)


def test_env_overrides_bucket(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "my-bucket")
    assert load_config()["data"]["gcs"]["bucket"] == "my-bucket"
