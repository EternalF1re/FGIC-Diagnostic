"""Manifest-content data identity wrapper for the final authorization gate."""
from __future__ import annotations

import json

import final_authorization_gate as base


def manifest_hashes(dataset: str) -> tuple[str, str]:
    path = base.ROOT / "manifests" / f"{dataset}_fold_and_class_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_count = 5994 if dataset == "cub" else 8144
    expected_classes = 200 if dataset == "cub" else 196
    if payload["dataset"] != dataset or payload["development_count"] != expected_count:
        raise RuntimeError(f"{dataset} manifest identity/count mismatch")
    if len(payload["assignments"]) != expected_count or len(payload["class_map"]) != expected_classes:
        raise RuntimeError(f"{dataset} assignment/class-map count mismatch")
    recomputed_fold = base.core.canonical_sha(payload["assignments"])
    recomputed_class = base.core.canonical_sha(payload["class_map"])
    if recomputed_fold != payload["fold_manifest_sha256"] or recomputed_class != payload["class_map_sha256"]:
        raise RuntimeError(f"{dataset} frozen manifest content hash mismatch")
    return recomputed_fold, recomputed_class


def main() -> None:
    base.live_fold_hashes = manifest_hashes
    base.main()
    authorization_path = base.ROOT / "FORMAL_RUN_AUTHORIZATION.json"
    payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    payload["data_identity_audit_mode"] = "recomputed content hashes and counts from frozen fold/class manifests; per-job DataLoader validates actual paths"
    authorization_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
