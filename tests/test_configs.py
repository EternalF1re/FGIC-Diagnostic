import json
from pathlib import Path


def test_all_public_json_files_parse():
    root = Path(__file__).resolve().parents[1]
    files = list((root / "configs").rglob("*.json"))
    assert files
    for path in files:
        json.loads(path.read_text(encoding="utf-8"))


def test_no_absolute_machine_path_in_maintained_config():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs").rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "D:/" not in text
        assert "D:\\" not in text
        assert "/mnt/" not in text

