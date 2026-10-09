import pytest

from ble_agent.experiment import validation_summary


def test_validation_reader_refuses_test_experiment(tmp_path, monkeypatch):
    import ble_agent.experiment as experiment

    monkeypatch.setattr(experiment, "experiment_path", lambda _: tmp_path / "test.json")
    (tmp_path / "test.json").write_text(
        '{"experiment_id":"test","split":"test","method":{},"metrics":{},"slices":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation"):
        validation_summary("test")
