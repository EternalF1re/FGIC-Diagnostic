"""Fail-closed identity gate between the frozen Windows snapshot and WSL2."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import timm
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARTIFACT_SHA = "773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb"
EXPECTED_BACKBONE_SHA = "45179af731d5ea5470ba5906fadf0b55b884338ca64d2b7f53f46096665ec6c6"
EXPECTED_CAL_COMMIT = "0ba9d5084f2532eeb21c9ef051c23f8b339595ff"
ARTIFACT = Path("<LOCAL_PATH> Face/模型数据/10118/huggingface/hub/models--timm--resnet50.a1_in1k/blobs/773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    plan = json.loads((ROOT / "CONTROLLED_REIMPLEMENTATION_60_JOB_PLAN.json").read_text(encoding="utf-8"))
    source_lock = json.loads((ROOT / "CONTROLLED_REIMPLEMENTATION_SOURCE_LOCK_MANIFEST.json").read_text(encoding="utf-8"))
    jobs = {(row["dataset"], row["fold"]): row for row in plan["jobs"] if row["method"] == "cal"}
    configs = {dataset: ROOT / "configs" / f"cal_{dataset}.json" for dataset in ("cub", "cars")}

    model = timm.create_model("resnet50", pretrained=False, num_classes=1000)
    model.load_state_dict(load_file(str(ARTIFACT), device="cpu"), strict=True)
    model.reset_classifier(0)
    model.eval()
    loaded_sha = state_sha(model.state_dict())
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 299, 299)
        final_feature = model.forward_features(dummy)
        pooled = model(dummy)

    checks = {
        "timm_version_1_0_21": timm.__version__ == "1.0.21",
        "cal_source_commit": all(job["source_commit"] == EXPECTED_CAL_COMMIT for job in jobs.values()),
        "cal_implementation_sha_matches_source_lock": file_sha(ROOT / "methods" / "cal" / "model.py") == source_lock["implementation_files_sha256"]["methods/cal/model.py"],
        "cub_config_sha": file_sha(configs["cub"]) == jobs[("cub", 0)]["config_sha256"],
        "cars_config_sha": file_sha(configs["cars"]) == jobs[("cars", 0)]["config_sha256"],
        "pretrained_artifact_sha": file_sha(ARTIFACT) == EXPECTED_ARTIFACT_SHA == source_lock["pretrained"]["artifact_sha256"],
        "loaded_initial_backbone_state_sha": loaded_sha == EXPECTED_BACKBONE_SHA == source_lock["pretrained"]["loaded_backbone_state_dict_sha256"],
        "cub_fold_manifest_sha": jobs[("cub", 0)]["fold_manifest_sha256"] == source_lock["fold_manifest_sha256"]["cub"],
        "cars_fold_manifest_sha": jobs[("cars", 0)]["fold_manifest_sha256"] == source_lock["fold_manifest_sha256"]["cars"],
        "cub_class_map_sha": jobs[("cub", 0)]["class_map_sha256"] == source_lock["class_map_sha256"]["cub"],
        "cars_class_map_sha": jobs[("cars", 0)]["class_map_sha256"] == source_lock["class_map_sha256"]["cars"],
        "augmentation_identity": all(json.loads(path.read_text(encoding="utf-8"))["training"]["augmentation_id"] == "cal_official_train_299_v1" for path in configs.values()),
        "evaluation_protocol_identity": len({jobs[(dataset, 0)]["evaluation_protocol_identifier"] for dataset in ("cub", "cars")}) == 1 and jobs[("cub", 0)]["evaluation_protocol_identifier"] == "pooled_oof_original_view_no_tta_no_fold_ensemble_v1",
        "dummy_final_feature_shape": list(final_feature.shape) == [1, 2048, 10, 10],
        "dummy_pooled_shape": list(pooled.shape) == [1, 2048],
    }
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "identities": {
            "cal_source_commit": EXPECTED_CAL_COMMIT,
            "cal_implementation_sha256": file_sha(ROOT / "methods" / "cal" / "model.py"),
            "cub_config_sha256": file_sha(configs["cub"]),
            "cars_config_sha256": file_sha(configs["cars"]),
            "pretrained_artifact_sha256": file_sha(ARTIFACT),
            "loaded_backbone_state_sha256": loaded_sha,
            "fold_manifest_sha256": source_lock["fold_manifest_sha256"],
            "class_map_sha256": source_lock["class_map_sha256"],
            "augmentation_manifest_sha256": file_sha(ROOT / "CONTROLLED_REIMPLEMENTATION_AUGMENTATION_MANIFEST.md"),
            "augmentation_id": "cal_official_train_299_v1",
            "evaluation_protocol_identifier": jobs[("cub", 0)]["evaluation_protocol_identifier"],
            "dummy_final_feature_shape": list(final_feature.shape),
            "dummy_pooled_shape": list(pooled.shape),
        },
    }
    output = ROOT / "preflight" / "wsl2_final" / "cal_protocol_identity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
