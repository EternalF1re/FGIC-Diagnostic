import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_path = Path(__file__).resolve().parents[1] / "scripts/verify_published_results.py"
_spec = importlib.util.spec_from_file_location("published_verify", _path)
verify = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = verify
_spec.loader.exec_module(verify)


def make_oof(tmp_path, **replace):
    data = {
        "sample_id": ["a", "b", "c", "d", "e"],
        "fold": [0, 1, 2, 3, 4],
        "y_true": [0, 1, 0, 1, 0],
        "y_pred": [0, 0, 0, 1, 1],
    }
    data.update(replace)
    path = tmp_path / "oof.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def test_metrics_recomputed_from_predictions(tmp_path):
    frame = verify.read_oof(make_oof(tmp_path), 5, 2)
    assert verify.metrics(frame)["accuracy"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "replace",
    [
        {"sample_id": ["a", "a", "c", "d", "e"]},
        {"fold": [0, 1, 2, 3, 6]},
        {"y_pred": [0, 0, 0, 1, 2]},
        {"y_true": [0, 1, 0, 1, -1]},
        {"fold": [0, 1, 2, 3, 3]},
    ],
)
def test_invalid_oof_rejected(tmp_path, replace):
    with pytest.raises(ValueError):
        verify.read_oof(make_oof(tmp_path, **replace), 5, 2)


def test_missing_sample_rejected(tmp_path):
    with pytest.raises(ValueError):
        verify.read_oof(make_oof(tmp_path), 6, 2)


def test_pair_aligns_ids_but_rejects_fold_or_label_mismatch(tmp_path):
    a = verify.read_oof(make_oof(tmp_path), 5, 2)
    b = a.iloc[::-1].copy()
    assert verify.align_pair(a, b)[0].sample_id.tolist() == sorted(a.sample_id.tolist())
    b.loc[b.index[0], "fold"] = 1
    with pytest.raises(ValueError):
        verify.align_pair(a, b)


def test_known_exact_mcnemar_and_candidate_minus_reference(tmp_path):
    a = verify.read_oof(make_oof(tmp_path), 5, 2)
    b = a.copy()
    b["y_pred"] = b.y_true
    result = verify.paired(a, b, "delta3", 42, 1000)
    assert result["delta_accuracy_pp"] == pytest.approx(40.0)
    assert result["mcnemar_p"] == pytest.approx(0.5)
    assert result["n10"] == 0 and result["n01"] == 2


@pytest.mark.parametrize("path", ["../outside.csv", "/tmp/outside.csv", "C:/Users/private.csv"])
def test_manifest_path_cannot_escape(tmp_path, path):
    with pytest.raises(ValueError):
        verify.safe_path(tmp_path, path)


def test_changed_manifest_bytes_rejected(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("original")
    digest = verify.sha256(path)
    path.write_text("changed")
    with pytest.raises(ValueError):
        verify.check_hash(tmp_path, "test.txt", digest)


def test_pair_preserves_archived_reference_order(tmp_path):
    a = verify.read_oof(make_oof(tmp_path), 5, 2).iloc[[3, 1, 4, 0, 2]]
    first, second = verify.align_pair(a, a.iloc[::-1])
    assert first.sample_id.tolist() == a.sample_id.tolist()
    assert second.sample_id.tolist() == a.sample_id.tolist()


def test_config_reference_checks_original_not_redacted_hash():
    manifest = pd.DataFrame([{"path": "config.json", "source_sha256": "original", "sha256": "redacted"}])
    index = pd.DataFrame([{"public_config": "config.json", "config_source_sha256": "original"}])
    assert verify.check_config_references(index, manifest) == 1
    index.loc[0, "config_source_sha256"] = "redacted"
    with pytest.raises(ValueError, match="Original configuration hash"):
        verify.check_config_references(index, manifest)


def test_archived_assertion_must_match_source_table(tmp_path):
    pd.DataFrame({"accuracy": [0.6]}).to_csv(tmp_path / "summary.csv", index=False)
    assertions = pd.DataFrame(
        [{"source_table": "summary.csv", "source_row": 0, "source_column": "accuracy", "value": 0.7}]
    )
    with pytest.raises(ValueError, match="Archived assertion"):
        verify.check_assertion_sources(tmp_path, assertions)
