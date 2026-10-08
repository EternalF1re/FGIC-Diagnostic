"""Catch configuration hash drift introduced by Git newline conversion."""

import csv
import hashlib
import subprocess
from pathlib import Path

import pytest


def test_git_index_config_bytes_match_documented_hashes():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("Archive has no Git index; artifact verifier checks extracted bytes instead")
    with (root / "docs/CONFIG_SHA256.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            indexed = subprocess.check_output(["git", "show", ":" + row["config_path"]], cwd=root)
            assert hashlib.sha256(indexed).hexdigest() == row["sha256"], row["config_path"]
